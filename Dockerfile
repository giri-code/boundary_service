FROM python:3.11-slim

# ── Labels ───────────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="boundary-service"
LABEL org.opencontainers.image.description="Object Boundary Detection Microservice"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

# ── System dependencies for OpenCV ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (cached layer) ───────────────────────────────────────
# 1. Install PyTorch explicitly from the CPU-only registry (prevents 2GB+ of NVIDIA CUDA bloat)
# Pinned to match requirements.txt — bump both together.
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 numpy==1.26.4 \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the lightweight application dependencies
COPY requirements.txt .
COPY vendor/ vendor/
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Download Weights ─────────────────────────────────────────────────────────
RUN python scripts/download_weights.py

# ── Runtime user (non-root for security) ─────────────────────────────────────
ENV HOME=/app
RUN useradd --system --no-create-home appuser \
    && mkdir -p /app/storage /app/models \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# ── CPU PyTorch Concurrency Optimization ─────────────────────────────────────
# For CPU-bound PyTorch/MobileSAM containers, 1 worker process prevents duplicate
# model weight memory multiplication (~400MB vs ~1.6GB+). Scale horizontally via Railway replicas.
ENV WEB_CONCURRENCY=1

CMD gunicorn app.main:app \
        --worker-class uvicorn.workers.UvicornWorker \
        --workers ${WEB_CONCURRENCY} \
        --bind 0.0.0.0:8000 \
        --timeout 120 \
        --keep-alive 5 \
        --access-logfile - \
        --error-logfile -
