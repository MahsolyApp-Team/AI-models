FROM python:3.11-slim

# Hugging Face Spaces runs as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy app files and model artifacts
COPY --chown=user app.py .
COPY --chown=user rf_model.pkl .
COPY --chown=user label_encoder.pkl .
COPY --chown=user feature_names.pkl .

# Hugging Face Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]