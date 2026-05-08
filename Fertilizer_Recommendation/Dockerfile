# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# HuggingFace Spaces runs containers as a non-root user (uid 1000).
# Create the user early so all subsequent layers have correct ownership.
RUN useradd -m -u 1000 appuser

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY app.py .

# ── Model artifacts ───────────────────────────────────────────────────────────
# The model/ folder must contain:
#   rf_model.pkl, scaler.pkl, label_encoder.pkl, column_names.pkl
COPY model/ ./model/

# ── Permissions ───────────────────────────────────────────────────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ── Runtime ───────────────────────────────────────────────────────────────────
# HuggingFace Spaces expects the app to listen on port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
