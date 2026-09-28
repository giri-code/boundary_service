"""Application and ML inference constants.

Fixed architectural invariants — same across all envs, so code const, not .env.
"""

# Maximum concurrent inference requests running in the AnyIO worker threadpool.
# Limits peak transient RAM allocation and prevents burst OOM crashes.
MAX_CONCURRENT_INFERENCE_THREADS: int = 4
