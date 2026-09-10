import time
import os
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
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


def _get_cached_memory_mb() -> float:
    """Return process memory, refreshed at most once per HEALTH_CACHE_TTL_SECONDS.

    FIX PERF-4: avoids a psutil syscall on every Kubernetes health probe.
    """
    now = time.monotonic()
    if now - _health_memory_cache["ts"] >= settings.HEALTH_CACHE_TTL_SECONDS:
        _health_memory_cache["value"] = get_process_memory_mb()
        _health_memory_cache["ts"] = now
    return _health_memory_cache["value"]


# ── Lifespan handler ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle using modern FastAPI lifespan context manager."""
    import torch
    # Prevent CPU oversubscription / thrashing on containerized instances
    torch.set_num_threads(2)

    logger.info("=" * 50)
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Environment : {settings.ENVIRONMENT}  Debug: {settings.DEBUG}")
    logger.info(f"Model       : {settings.SEGMENTATION_MODEL_PROVIDER}")
    logger.info(f"Storage     : {settings.STORAGE_BACKEND}")
    logger.info(f"Memory      : {get_process_memory_mb():.1f} MB")
    logger.info("=" * 50)

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
def _run_boundary_detection(request: BoundaryRequest) -> BoundaryResponse:
    """Synchronous boundary detection logic, safe to run in a threadpool worker.

    BUG-2 FIX: this function handles all blocking I/O (disk / S3 / GCS reads)
    and CPU-bound inference. FastAPI's run_in_threadpool dispatches it off the
    async event loop so it never blocks other concurrent async requests.
    """
    start_time = time.time()

    if not request.image_path and not request.photo_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either image_path or photo_id",
        )

    embedding_dict = None
    w, h = 0, 0
    if request.photo_id:
        embedding_dict = EmbeddingService.load(request.photo_id)
        if embedding_dict is None:
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
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        except (ValueError, PermissionError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Storage retrieval error: {exc}",
            )

    # Validate click coordinates (if we have image dimensions)
    if w > 0 and h > 0:
        if not (0 <= request.x < w) or not (0 <= request.y < h):
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
        logger.error(f"Segmentation error ({model_engine.name}): {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation model error ({model_engine.name}): {exc}",
        )

    # 5. Extract simplified polygon boundary and inner holes
    tolerance = (
        request.tolerance
        if request.tolerance is not None
        else settings.DEFAULT_POLYGON_TOLERANCE
    )
    contour_data = ContourService.process_mask(binary_mask, tolerance=tolerance)

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

    - **image_path**: local path, relative filename, `s3://`, `gs://`, or `http://` URL
    - **x** / **y**: click coordinates in pixel space
    - **tolerance**: RDP simplification factor (0.0001–0.1, default 0.005)
    - **model_provider**: deprecated, ignored — MobileSAM-only service.
    """
    # FIX BUG-2: run blocking I/O + CPU inference in threadpool, not on event loop
    return await run_in_threadpool(_run_boundary_detection, request)


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
        return {"success": True, "message": "Embedding deleted"}
    except Exception as exc:
        logger.error(f"Failed to delete embedding for {photo_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

def _run_encoding(request: EncodeRequest) -> dict:
    from .services.embedding_service import EmbeddingService
    if EmbeddingService.load(request.photo_id) is not None:
        logger.info(f"Embedding already exists for {request.photo_id}. Skipping calculation.")
        return {"success": True, "message": "Embedding already exists"}

    try:
        image = StorageProviderFactory.read_image(request.image_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage retrieval error: {exc}",
        )

    model_engine = SegmentationModelFactory.get_engine("mobile_sam")
    if not hasattr(model_engine, "encode_image"):
        raise HTTPException(
            status_code=500, detail="Engine does not support separate encoding"
        )

    try:
        embedding_dict = model_engine.encode_image(image)
        EmbeddingService.save(request.photo_id, embedding_dict)
    except Exception as exc:
        logger.error(f"Encoding error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Encoding error: {exc}",
        )
    return {"success": True, "photo_id": request.photo_id}


@app.post(
    "/api/v1/boundary/encode",
    tags=["Boundary Detection"],
    summary="Pre-calculate embeddings for an image",
    dependencies=[Depends(verify_internal_token)],
)
async def encode_image_endpoint(request: EncodeRequest):
    return await run_in_threadpool(_run_encoding, request)
