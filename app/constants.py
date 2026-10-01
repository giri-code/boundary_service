"""Application and ML inference constants.

Fixed architectural invariants — same across all envs, so code const, not .env.
"""

# Maximum concurrent inference requests admitted to the worker threadpool.
# Single-box tuning (1-2G): keeps click bursts from combining transients
# into an OOM. Per-click cost (~15MB, ~33ms) supports high RPS per replica.
MAX_CONCURRENT_INFERENCE_THREADS: int = 10

# Maximum concurrent encode (ViT embedding) requests. Separate limiter from
# inference (bulkhead): bulk encode batches must never starve user-facing
# click inference. One 24MP encode transients ~500MB, so this gate is the
# memory bound for the encode lane — size it against the box, not the SLO.
# Single box: 1 serializes ViT encodes (2 fit only on a 2G box).
MAX_CONCURRENT_ENCODE_THREADS: int = 3

# Maximum concurrent inference + encode executions COMBINED. The default
# thread pool is sized to this same number (see lifespan), so admission
# (per-op → overall) and threads agree: no hidden serialization below,
# no unbounded threads above. Total transient budget ≈ overall × per-request
# peak — keep this consistent with the container memory limit.
MAX_CONCURRENT_TOTAL_THREADS: int = 14

# Storage precision for photo embeddings:
# - "float32": 4.2 MB / photo (baseline, uncompressed)
# - "float16": 2.1 MB / photo (50% smaller, lossless)
# - "int8":    1.0 MB / photo (75% smaller, high quality)
# - "int4":    512 KB / photo (88% smaller, compact)
EMBEDDING_PRECISION: str = "int4"

