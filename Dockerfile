# SF 311 Triage API (CPU only).
#   docker build -t sf311-triage .
#   docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" sf311-triage
# Without a models/ mount the API serves the labelled stub predictors.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    PYTHONPATH=/app/src:/app \
    SF311_MODELS_DIR=/app/models

WORKDIR /app

# libgomp is needed by LightGBM.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first for layer caching.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install ".[ml]"

COPY app/ app/
RUN mkdir -p /app/models && useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
