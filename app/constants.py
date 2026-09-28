"""Application and ML inference constants.

Fixed architectural invariants — same across all envs, so code const, not .env.
"""

# Maximum concurrent inference requests running in the AnyIO worker threadpool.
# Limits peak transient RAM allocation and prevents burst OOM crashes.
MAX_CONCURRENT_INFERENCE_THREADS: int = 4

# Storage precision for photo embeddings:
# - "float32": 4.2 MB / photo (baseline, uncompressed)
# - "float16": 2.1 MB / photo (50% smaller, lossless)
# - "int8":    1.0 MB / photo (75% smaller, high quality)
# - "int4":    512 KB / photo (88% smaller, compact)
EMBEDDING_PRECISION: str = "int4"

