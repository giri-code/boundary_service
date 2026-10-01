import time
import os
import threading
import warnings
from contextlib import asynccontextmanager

# Suppress harmless timm & MobileSAM registry overwrite warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
    Depends,
)
from fastapi.concurrency import run_in_threadpool
from anyio.to_thread import run_sync as run_in_limited_thread
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .constants import (
    MAX_CONCURRENT_INFERENCE_THREADS,
    MAX_CONCURRENT_ENCODE_THREADS,
    MAX_CONCURRENT_TOTAL_THREADS,
)
from .security import verify_internal_token
from .schemas.boundary import (
    BoundaryRequest,
    BoundaryResponse,
    ImageDimensions,
    EncodeRequest,
)
from .storage.factory import StorageProviderFactory
from .models.factory import SegmentationModelFactory
from .services.contour_service import ContourService
from .utils.logger import logger, RequestTimingMiddleware
from .utils.memory import get_process_memory_mb, force_garbage_collection
from .services.embedding_service import EmbeddingService

# ── Cached health memory reading ──────────────────────────────────────────────
_health_memory_cache: dict = {"value": 0.0, "ts": 0.0}
_health_memory_lock = threading.Lock()

# Per-operation AnyIO capacity limiters (bulkhead): inference (user-facing
# clicks, tight SLO) and encode (bulk ViT embedding, slower SLO) draw from
# the same worker threadpool but are admitted separately, so a bulk encode
# batch can never occupy more than its share and starve clicks. The overall
# limiter caps inference + encode COMBINED so total transient memory never
# exceeds the validated envelope.
# Created in lifespan (async context); None only if lifespan never ran
# (endpoints then fall back to the default pool, unthrottled).
_inference_limiter = None
_encode_limiter = None
_overall_limiter = None


async def _run_in_bulkhead(func, arg, op_limiter):
    """Run blocking func in the worker threadpool under both gates.

    Acquisition order is always per-op FIRST, overall second (never reversed).
    Waiters pile on the per-op gate WITHOUT holding overall tickets, so queued
    encodes can never starve clicks by occupying the combined budget. (The reverse
    order — hold overall while parked on the op gate — deadlocks the bulkhead's
    purpose under bursty encode load.) Either limiter being None (lifespan never
    ran, e.g. unit tests bypassing it) degrades to the remaining gate.

    Note: once the op gate is held via `async with`, threads come from the default
    pool (passing the same limiter to run_sync as well would self-deadlock, since
    CapacityLimiter is not reentrant).
    """
    if op_limiter is None and _overall_limiter is None:
        return await run_in_limited_thread(func, arg)
    if op_limiter is None:
        async with _overall_limiter:
            return await run_in_limited_thread(func, arg)
    if _overall_limiter is None:
        return await run_in_limited_thread(func, arg, limiter=op_limiter)
    async with op_limiter:
        async with _overall_limiter:
            return await run_in_limited_thread(func, arg)


def _get_cached_memory_mb() -> float:
    """Return process memory, refreshed at most once per HEALTH_CACHE_TTL_SECONDS.

    FIX PERF-4: avoids a psutil syscall on every Kubernetes health probe.
    """
    now = time.monotonic()
    with _health_memory_lock:
        if now - _health_memory_cache["ts"] >= settings.HEALTH_CACHE_TTL_SECONDS:
            _health_memory_cache["value"] = get_process_memory_mb()
            _health_memory_cache["ts"] = now
        return _health_memory_cache["value"]


# ── Lifespan handler ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle using modern FastAPI lifespan context manager."""
    global _inference_limiter, _encode_limiter, _overall_limiter

    import torch
    # Prevent CPU oversubscription / thrashing on containerized instances
    torch.set_num_threads(2)
    try:
        import cv2

        # OpenCV runs its own all-core threadpool: cap it like torch, or 2 concurrent
        # requests × 2 OMP threads × uncapped cv2 oversubscribe small containers.
        cv2.setNumThreads(2)
    except Exception as exc:
        logger.warning(f"Could not cap OpenCV threads: {exc}")

    # Cap the AnyIO worker threadpool limiter to prevent concurrent OOM spikes & CPU thrashing
    try:
        import anyio
        from anyio import CapacityLimiter

        limiter = anyio.to_thread.current_default_thread_limiter()
        # The default pool is the actual thread source for _run_in_bulkhead (op gates
        # are admission-only via `async with` — passing the same limiter to run_sync
        # as well would self-deadlock, CapacityLimiter is not reentrant). Size it at
        # the COMBINED total: lower would serialize ops against each other and
        # neutralize the bulkhead; the per-op gates provide the fairness.
        limiter.total_tokens = MAX_CONCURRENT_TOTAL_THREADS
        _inference_limiter = CapacityLimiter(MAX_CONCURRENT_INFERENCE_THREADS)
        _encode_limiter = CapacityLimiter(MAX_CONCURRENT_ENCODE_THREADS)
        _overall_limiter = CapacityLimiter(MAX_CONCURRENT_TOTAL_THREADS)
        logger.info(
            f"AnyIO default thread pool capped at {limiter.total_tokens} total threads "
            "(thread source for the bulkhead). "
            f"Per-op bulkheads: inference={MAX_CONCURRENT_INFERENCE_THREADS}, "
            f"encode={MAX_CONCURRENT_ENCODE_THREADS}, "
            f"total={MAX_CONCURRENT_TOTAL_THREADS}."
        )
    except Exception as exc:
        logger.warning(f"Could not configure AnyIO thread limiter: {exc}")

    logger.info(
        f"Starting {settings.APP_NAME} v{settings.VERSION} "
        f"[env={settings.ENVIRONMENT}, debug={settings.DEBUG}, model={settings.SEGMENTATION_MODEL_PROVIDER}, "
        f"storage={settings.STORAGE_BACKEND}, memory={get_process_memory_mb():.1f}MB]"
    )

    try:
        web_concurrency = int(os.getenv("WEB_CONCURRENCY", "1"))
        if web_concurrency > 1:
            # Fail fast: per-process gates (bulkhead limiters, in-memory caches) and
            # per-process ViT transients make N workers N× the memory profile the box
            # was sized for. Scale via container replicas, never workers.
            raise RuntimeError(
                f"WEB_CONCURRENCY={web_concurrency} refused: ML inference containers must "
                "run with 1 worker process per container (WEB_CONCURRENCY=1) and scale "
                "horizontally via container replicas."
            )
    except ValueError:
        pass


    # MobileSAM-only service: weights must be present. Fail fast here so a
    # missing/truncated checkpoint crashes the container (visible restart +
    # alert) instead of serving 500s on every encode/click.
    checkpoint = settings.MOBILE_SAM_CHECKPOINT
    if not os.path.exists(checkpoint) or os.path.getsize(checkpoint) < 1_000_000:
        raise RuntimeError(
            f"MobileSAM checkpoint missing or truncated at: {checkpoint}. "
            "Run 'python scripts/download_weights.py' or fix MOBILE_SAM_CHECKPOINT."
        )

    # Pre-warm MobileSAM model weights asynchronously in background pool so startup doesn't stall
    try:
        engine = SegmentationModelFactory.get_engine("mobile_sam")
        if hasattr(engine, "_load_model"):
            await run_in_threadpool(engine._load_model)
            logger.info("MobileSAM pre-warmed successfully at startup.")
    except Exception as exc:
        logger.error(f"MobileSAM startup pre-warm failed: {exc}")
        raise

    yield
    logger.info("Shutting down — running final GC...")
    force_garbage_collection()


# ── OpenAPI tag metadata ──────────────────────────────────────────────────────
tags_metadata = [
    {
        "name": "Boundary Detection",
        "description": (
            "Detect precise object polygon boundary coordinates and inner hole geometries "
            "from a single user click point (x, y) on an image."
        ),
    },
    {
        "name": "System Health",
        "description": "Health check, runtime metrics, active provider status.",
    },
]

# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Production-ready Object Boundary Detection Microservice. "
        "Extracts exact polygon boundary coordinates and inner holes for objects in photos "
        "based on user click points (x, y). Supports local storage, AWS S3, and GCP Cloud Storage."
    ),
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception handlers ────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code >= 500:
        logger.error(
            f"HTTP {exc.status_code} at {request.method} {request.url.path}: {exc.detail}"
        )
    elif exc.status_code >= 400:
        logger.warning(
            f"HTTP {exc.status_code} at {request.method} {request.url.path}: {exc.detail}"
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "path": str(request.url.path),
        },
    )



@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error at {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal Server Error",
            "path": str(request.url.path),
        },
    )


# ── System endpoints ──────────────────────────────────────────────────────────
@app.get("/", tags=["System Health"])
def root():
    """Root endpoint — returns API metadata and active provider information."""
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "active_model_provider": settings.SEGMENTATION_MODEL_PROVIDER,
        "active_storage_backend": settings.STORAGE_BACKEND,
    }


@app.get("/health", tags=["System Health"])
def health_check():
    """Lightweight health probe for Kubernetes, GCP Cloud Run, and AWS ECS.

    Memory reading is cached (TTL = HEALTH_CACHE_TTL_SECONDS) to avoid a
    psutil syscall on every frequent load-balancer probe.
    `model_loaded`/`checkpoint_bytes` are file-backed signals only — they never
    trigger a model load, so probes stay cheap.
    """
    try:
        checkpoint_bytes = os.path.getsize(settings.MOBILE_SAM_CHECKPOINT)
    except OSError:
        checkpoint_bytes = 0
    try:
        engine = SegmentationModelFactory.get_engine("mobile_sam")
        model_loaded = getattr(engine, "_sam_model", None) is not None
    except Exception:
        model_loaded = False
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "model_provider": settings.SEGMENTATION_MODEL_PROVIDER,
        "storage_backend": settings.STORAGE_BACKEND,
        "memory_mb": round(_get_cached_memory_mb(), 2),
        "model_loaded": model_loaded,
        "checkpoint_bytes": checkpoint_bytes,
    }


# ── Core boundary detection ───────────────────────────────────────────────────
def _rescale_contour_to_original(contour_data: dict, sx: float, sy: float) -> dict:
    """Scale decode-resolution polygons back to original image pixels."""
    from .schemas.boundary import Point, BoundingBox

    def _scale_pts(pts):
        return [
            Point(x=int(round(p.x * sx)), y=int(round(p.y * sy))) for p in pts
        ]

    bb = contour_data["bounding_box"]
    return {
        "outer_boundary": _scale_pts(contour_data["outer_boundary"]),
        "holes": [_scale_pts(hole) for hole in contour_data["holes"]],
        "bounding_box": BoundingBox(
            xmin=int(round(bb.xmin * sx)),
            ymin=int(round(bb.ymin * sy)),
            xmax=int(round(bb.xmax * sx)),
            ymax=int(round(bb.ymax * sy)),
        ),
        "area_pixels": int(round(contour_data["area_pixels"] * sx * sy)),
    }


def _run_boundary_detection(request: BoundaryRequest) -> BoundaryResponse:
    """Synchronous boundary detection logic, safe to run in a threadpool worker.

    BUG-2 FIX: this function handles all blocking I/O (disk / S3 / GCS reads)
    and CPU-bound inference. FastAPI's run_in_threadpool dispatches it off the
    async event loop so it never blocks other concurrent async requests.
    """
    start_time = time.time()

    if not request.image_path and not request.photo_id:
        logger.warning("Boundary request rejected: neither image_path nor photo_id provided.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either image_path or photo_id",
        )

    embedding_dict = None
    w, h = 0, 0
    if request.photo_id:
        embedding_dict = EmbeddingService.load(request.photo_id)
        if embedding_dict is None:
            logger.warning(
                f"Embedding not ready for photo_id='{request.photo_id}'. Boundary calculation rejected with 409 Conflict."
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Embedding not ready for photo_id: '{request.photo_id}'. Background encoding is in progress.",
            )

    image = None
    if embedding_dict:
        # We have the embedding, no need to read the image from disk!
        h, w = embedding_dict["original_size"]
    elif request.image_path:
        # Fallback: Read image from storage provider
        try:
            image = StorageProviderFactory.read_image(request.image_path)
            h, w = image.shape[:2]
        except FileNotFoundError as exc:
            logger.warning(f"Image not found at path '{request.image_path}': {exc}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        except (ValueError, PermissionError) as exc:
            logger.warning(f"Invalid image or permission error for path '{request.image_path}': {exc}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            logger.error(
                f"Storage retrieval error reading image from '{request.image_path}': {exc}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Storage retrieval error: {exc}",
            )

    # Validate click coordinates (if we have image dimensions)
    if w > 0 and h > 0:
        if not (0 <= request.x < w) or not (0 <= request.y < h):
            logger.warning(
                f"Click point ({request.x}, {request.y}) out of bounds ({w}×{h}) "
                f"for photo_id='{request.photo_id}', image_path='{request.image_path}'"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Click point ({request.x}, {request.y}) is outside "
                    f"image bounds ({w}×{h})."
                ),
            )

    # MobileSAM-only service: provider override is ignored (kept in schema for
    # backward compatibility). OpenCV is not reachable via the API.
    model_engine = SegmentationModelFactory.get_engine("mobile_sam")

    # Predict binary object mask
    try:
        if embedding_dict:
            binary_mask, confidence = model_engine.predict_from_embedding(
                embedding_dict=embedding_dict,
                point=(request.x, request.y),
                level=request.level,
            )
        else:
            if image is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Image not found and embedding not available",
                )
            binary_mask, confidence = model_engine.predict_mask(
                image=image,
                point=(request.x, request.y),
                cache_key=request.image_path,
                level=request.level,
            )
    except Exception as exc:
        logger.error(
            f"Segmentation inference error ({model_engine.name}) for photo_id='{request.photo_id}', "
            f"image_path='{request.image_path}', point=({request.x}, {request.y}), level={request.level}: {exc}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation model error ({model_engine.name}): {exc}",
        )

    # Free per-request embedding/image refs before contour: shared Redis/S3
    # caches are untouched, but this drops ~4MB (+H*W*3 image) ahead of the +2HW
    # contour peak within the same thread.
    embedding_dict = None
    image = None

    # 5. Extract simplified polygon boundary and inner holes.
    # The MobileSAM mask may be at capped decode resolution (long side ≤
    # MAX_DECODE_DIMENSION): trace in mask space, then scale polygons back to
    # original pixels. Small photos decode at full res (scale == 1, no-op).
    mh, mw = binary_mask.shape[:2]
    sx = (w / mw) if mw else 1.0
    sy = (h / mh) if mh else 1.0
    mask_click = (
        (request.x / sx, request.y / sy)
        if (sx != 1.0 or sy != 1.0)
        else (request.x, request.y)
    )
    tolerance = (
        request.tolerance
        if request.tolerance is not None
        else settings.DEFAULT_POLYGON_TOLERANCE
    )
    contour_data = ContourService.process_mask(
        binary_mask,
        tolerance=tolerance,
        click_point=mask_click,
    )
    if sx != 1.0 or sy != 1.0:
        contour_data = _rescale_contour_to_original(contour_data, sx, sy)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return BoundaryResponse(
        success=True,
        outer_boundary=contour_data["outer_boundary"],
        holes=contour_data["holes"],
        bounding_box=contour_data["bounding_box"],
        area_pixels=contour_data["area_pixels"],
        confidence_score=round(confidence, 4),
        image_dimensions=ImageDimensions(width=w, height=h),
        model_used=model_engine.name,
        processing_time_ms=elapsed_ms,
    )


@app.post(
    "/api/v1/boundary",
    response_model=BoundaryResponse,
    tags=["Boundary Detection"],
    summary="Detect object boundary by click point",
    dependencies=[Depends(verify_internal_token)],
)
async def detect_object_boundary(request: BoundaryRequest):
    """Detect exact object boundary polygon from an image path and click point (x, y).

    - **image_path**: `s3://` URI or bucket-relative key (HTTP URLs rejected)
    - **x** / **y**: click coordinates in pixel space
    - **tolerance**: RDP simplification factor (0.0001–0.1, default 0.005)
    - **model_provider**: deprecated, ignored — MobileSAM-only service.
    """
    # FIX BUG-2: run blocking I/O + CPU inference in threadpool, not on event loop.
    # Bulkhead: inference admissions are capped by _inference_limiter (and the
    # overall total) so bulk encodes can never starve clicks.
    return await _run_in_bulkhead(
        _run_boundary_detection, request, _inference_limiter
    )


@app.delete(
    "/api/v1/boundary/embedding/{photo_id}",
    tags=["Boundary Detection"],
    summary="Delete a cached embedding for a photo",
    dependencies=[Depends(verify_internal_token)],
)
async def delete_embedding(photo_id: str):
    from .services.embedding_service import EmbeddingService
    try:
        EmbeddingService.delete(photo_id)
        # The in-process engine cache is keyed by image_path and capped at 15 entries:
        # a delete must not leave servable stale masks behind. Photo bytes are
        # immutable so this is belt-and-braces, not a hot path.
        try:
            SegmentationModelFactory.get_engine("mobile_sam").clear_cache()
        except Exception as exc:
            logger.warning(f"Engine cache clear failed for photo_id='{photo_id}': {exc}")
        return {"success": True, "message": "Embedding deleted"}
    except Exception as exc:
        logger.error(f"Failed to delete embedding for photo_id='{photo_id}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

def _run_encoding(request: EncodeRequest) -> dict:
    from .services.embedding_service import EmbeddingService, S3PersistenceError
    from .services.encode_claim import EncodeClaim, ClaimHeartbeat

    if EmbeddingService.load(request.photo_id) is not None:
        logger.info(f"Embedding already exists for photo_id='{request.photo_id}'. Skipping encoding.")
        return {"success": True, "message": "Embedding already exists"}

    # Claim BEFORE any image I/O (§5.6 ordering invariant): losers return 409
    # without downloading/decoding a single byte — never ViT transients.
    claim = EncodeClaim(EmbeddingService.get_redis())
    token = claim.acquire(request.photo_id)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Embedding not ready for photo_id: '{request.photo_id}'. Encoding in progress.",
        )
    try:
        # Double-check: a winner may have finished between our fast-path and claim.
        if EmbeddingService.load(request.photo_id) is not None:
            logger.info(f"Embedding appeared for photo_id='{request.photo_id}' during claim. Skipping encoding.")
            return {"success": True, "message": "Embedding already exists"}

        try:
            image = StorageProviderFactory.read_image(request.image_path)
        except FileNotFoundError as exc:
            # Deterministic failure: must NOT retry (BullMQ Unrecoverable path).
            logger.warning(f"Image not found for encoding photo_id='{request.photo_id}': {exc}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        except (ValueError, PermissionError) as exc:
            logger.warning(f"Invalid image for encoding photo_id='{request.photo_id}': {exc}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except Exception as exc:
            logger.error(
                f"Failed to read image for encoding photo_id='{request.photo_id}', image_path='{request.image_path}': {exc}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Storage retrieval error: {exc}",
            )

        model_engine = SegmentationModelFactory.get_engine("mobile_sam")
        if not hasattr(model_engine, "encode_image"):
            logger.error(f"Model engine '{model_engine.name}' does not support encode_image for photo_id='{request.photo_id}'")
            raise HTTPException(
                status_code=500, detail="Engine does not support separate encoding"
            )

        try:
            # Heartbeat covers the long pole (ViT encode, seconds on big images):
            # a live encode never loses its claim; a dead process's claim dies
            # with it at TTL expiry.
            with ClaimHeartbeat(claim, request.photo_id, token):
                embedding_dict = model_engine.encode_image(image)
                # Drop full-res image refs before save: the ViT transient is over,
                # don't carry ~150MB of uint8 into serialization.
                del image
                EmbeddingService.save(request.photo_id, embedding_dict)
            logger.info(f"Successfully encoded and stored embedding for photo_id='{request.photo_id}'")
        except S3PersistenceError as exc:
            # Durable truth missing: fail loud so BullMQ retries → DLQ, never a
            # false 200 that rots at Redis TTL expiry.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            )
        except Exception as exc:
            logger.error(
                f"Encoding error for photo_id='{request.photo_id}', image_path='{request.image_path}': {exc}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Encoding error: {exc}",
            )
        return {"success": True, "photo_id": request.photo_id}
    finally:
        claim.release(request.photo_id, token)



@app.post(
    "/api/v1/boundary/encode",
    tags=["Boundary Detection"],
    summary="Pre-calculate embeddings for an image",
    dependencies=[Depends(verify_internal_token)],
)
async def encode_image_endpoint(request: EncodeRequest):
    # Bulkhead: encodes are admitted via their own limiter (slower SLO,
    # fewer slots) plus the overall total, so bulk batches never starve
    # inference and total threads stay within the validated envelope.
    return await _run_in_bulkhead(_run_encoding, request, _encode_limiter)
