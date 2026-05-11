# ── Base image ────────────────────────────────────────────────────────────────
# HuggingFace Spaces expects the server on port 7860.
# Use slim Python 3.11 to keep the image small; install CPU-only PyTorch
# (Spaces free tier is CPU-only; switch the index URL for GPU tiers).
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (required by HuggingFace Spaces) ───────────────────────────
RUN useradd -m -u 1000 appuser
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .

# Install CPU-only PyTorch first (much smaller download than the default CUDA
# build). Remove the --index-url line if you are targeting a GPU Space.
RUN pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY main.py .

# ── Model weights ─────────────────────────────────────────────────────────────

COPY vgg19_rgb_model.pth    .

# ── Permissions ───────────────────────────────────────────────────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ── Runtime ───────────────────────────────────────────────────────────────────
# HuggingFace Spaces routes external traffic to port 7860.
EXPOSE 7860

ENV PYTHONUNBUFFERED=1 \
    RGB_WEIGHTS=vgg19_rgb_model.pth 

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
