# Serving image for the Vision RAG API *and* its UI (src/server.py + ui/).
#
# The React app is compiled by the ui-builder stage below and served by FastAPI itself,
# so a deployment is one container on one origin - no separate web server, and no CORS
# in the deployed shape. On Linux `uv sync` pulls the CUDA 12.8 torch wheels (they
# bundle their own CUDA libs), so the image is GPU-capable with `docker run --gpus all`
# and auto-falls back to CPU when no GPU is present (embedder._device_and_dtype).
#
#   docker build -t vision-rag .
#   docker run --rm -p 8000:8000 \
#     -e GEMINI_API_KEY=... -e QDRANT_URL=http://host.docker.internal:6333 \
#     -e API_KEY=...   \                                  # gate the API; see src/auth.py
#     -v vision-rag-hf:/home/appuser/.cache/huggingface \ # persist the model download
#     -v vision-rag-pages:/app/page_images \              # REQUIRED: half the corpus
#     -v vision-rag-pdfs:/app/pdfs \                      # the source documents
#     [--gpus all] vision-rag
#
# **page_images/ is not a cache.** It and the vectors in Qdrant are two halves of one
# corpus, so they must persist and be backed up together. Run without that mount and a
# container recreate keeps the vectors, loses the pages, and every query answers "not
# found" while /corpus still lists a full corpus. The server reports the split at boot
# and on /health; a plain re-ingest repairs it. (PAGE_IMAGES_DIR / PDFS_DIR relocate
# these if you would rather mount elsewhere; stored image paths are relative, so moving
# the directory is safe.)
#
# First boot downloads the ~2B ColQwen2 model from HuggingFace into the mounted cache;
# subsequent boots are warm. Qdrant must be reachable at QDRANT_URL.

# ---- ui builder: compile the React app to static assets ---------------------------
FROM node:22-slim AS ui-builder

WORKDIR /ui

# Manifests first, so editing a component doesn't re-resolve the npm tree.
COPY ui/package.json ui/package-lock.json ./
RUN npm ci

COPY ui/ ./
# `npm run build` is `tsc && vite build`, so a type error fails the image build rather
# than shipping a broken bundle. No VITE_API_BASE is set: a production build defaults
# to same-origin relative URLs, which is exactly what serving from FastAPI needs.
RUN npm run build

# ---- builder: resolve + install runtime deps into a venv --------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Copy bytecode-compiled, real files (not symlinks) so the venv survives the stage copy.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependency layer: only the manifests, so a code change doesn't re-resolve the (large)
# ML stack. --no-dev drops pytest/ruff/mypy; the project itself isn't a package (no
# build-system), so uv installs just the dependencies - src/ is added below.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/

# ---- runtime: slim image with just the venv, source, and poppler -------------------
FROM python:3.13-slim-bookworm

# poppler-utils backs pdf2image (page rendering); libgomp1 is torch's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user; HF_HOME under its home so the model cache is a mountable volume.
RUN useradd --create-home --uid 1000 appuser
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app
# --chown sets ownership as the layers are written; a later `chown -R /app` would
# instead re-copy the multi-GB venv into a new layer, doubling the image size.
COPY --chown=appuser:appuser --from=builder /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/
# The compiled UI, at the path config.UI_DIST_DIR expects (ROOT_DIR/ui/dist). Its
# presence is what makes server._mount_ui serve the app at "/".
COPY --chown=appuser:appuser --from=ui-builder /ui/dist ./ui/dist
# StaticFiles mounts page_images/ at startup, so the dir must exist even before ingest.
RUN mkdir -p /app/page_images/crops /app/pdfs \
    && chown appuser:appuser /app /app/page_images /app/page_images/crops /app/pdfs

USER appuser
EXPOSE 8000

# /health reports 503 until the model is loaded + Qdrant is reachable; the generous
# start-period covers the first-boot model download.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Single worker only (warm model + asyncio.Lock serialize the GPU) - never --workers >1.
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
