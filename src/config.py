"""Central configuration: env vars, model names, paths, and Qdrant settings."""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent.parent
PDFS_DIR = ROOT_DIR / "pdfs"
PAGE_IMAGES_DIR = ROOT_DIR / "page_images"
# Cropped/annotated chart slices produced at answer time. Under page_images/ but
# never re-ingested (ingest only globs pdfs/).
CROPS_DIR = PAGE_IMAGES_DIR / "crops"
# Qdrant server URL (e.g. http://localhost:6333). When unset, fall back to the
# embedded on-disk store at QDRANT_PATH - convenient for quick prototyping/tests
# with no container running. Point the app at the Dockerized server by setting
# QDRANT_URL in .env (see docker-compose.yml).
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # None for the local unauthenticated container
QDRANT_PATH = ROOT_DIR / "qdrant_data"        # embedded on-disk fallback store

# Directory containing poppler's CLI tools (pdftoppm/pdftocairo), used by pdf2image.
# Override via POPPLER_PATH; otherwise derive it from pdftoppm on PATH.
# None lets pdf2image fall back to PATH (works on Linux where poppler-utils is installed system-wide).
_poppler_bin = shutil.which("pdftoppm")
POPPLER_PATH = os.getenv("POPPLER_PATH") or (str(Path(_poppler_bin).parent) if _poppler_bin else None)

# colqwen2-v1.0 (~2B) fits comfortably in 8 GB VRAM.
# For higher chart/table accuracy on a bigger GPU, use "vidore/colqwen2.5-v0.2"
# (the embedder auto-selects the ColQwen2_5 loader for colqwen2.5 checkpoints).
# Env-overridable so a model A/B is a re-ingest, not a code edit.
COLPALI_MODEL = os.getenv("COLPALI_MODEL") or "vidore/colqwen2-v1.0"
# Page render resolution. Higher DPI = more ColQwen patches = finer detail on dense
# tables/small text, at more ingest time + storage. Env-overridable for A/B ingests.
RENDER_DPI = int(os.getenv("RENDER_DPI", "150"))
# Identifies what produced a stored vector. Written into every point's payload so an
# incremental ingest can tell a still-current page from a stale one: changing the model
# or the render DPI invalidates every embedding while leaving the PDF bytes untouched,
# which a content hash alone would miss (see src/ingest.py).
EMBED_VERSION = f"{COLPALI_MODEL}@{RENDER_DPI}"

COLLECTION_NAME = "pdf_pages"
VECTOR_DIM = 128 # ColQwen emits one 128-d vector per patch
# Both env-overridable for the same reason RENDER_DPI is: an eval arm should be a
# command-line prefix (`RERANK_K=3 uv run python eval/run_eval.py ...`), not a code
# edit that has to be reverted before the comparison run. load_dotenv() does not
# override an already-set variable, so the prefix wins over .env.
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "10"))  # candidate pages pulled from Qdrant per query
# 3, not 2: at 2 the reranker spent both slots inside one document on cross-document
# questions, and the answer step filled the missing half from parametric memory - with
# numbers that were plausible and *wrong* (BERT-base "108M" for 110M, BEIR "19 datasets"
# for 18). Measured over 83 questions: 3 genuine answer/citation fixes, zero regressions
# across 144 paired row comparisons, for +7.8% cost and flat latency. See the
# instrument-sharpening pass in PRODUCTION_HARDENING.md.
RERANK_K = int(os.getenv("RERANK_K", "3"))       # pages kept after the Gemini rerank pass
                                                 # (a cap when RERANK_ADAPTIVE)
# When true, rerank keeps a *variable* number of pages (1..RERANK_K) - only the
# pages the model judged relevant, instead of always topping up to RERANK_K. Trades
# a little predictability for answer precision. Off by default until an eval diff
# proves it wins (PRODUCTION_HARDENING.md retrieval-quality pass).
RERANK_ADAPTIVE = os.getenv("RERANK_ADAPTIVE", "false").strip().lower() in ("1", "true", "yes")
# Binary quantization is lossy; the fast quantized pass pulls RETRIEVE_K * this many
# candidates, then rescores them against the full-precision vectors on disk. Higher
# recovers recall@1 that quantization costs, at a little more disk I/O per query.
RESCORE_OVERSAMPLING = float(os.getenv("RESCORE_OVERSAMPLING", "2.0"))
UPSERT_BATCH_SIZE = 8    # pages per Qdrant upsert flush; small enough that a batch's
                         # multivector payload (~1.4 MB/page) stays well under Qdrant's
                         # REST size limit, even on the default 32 MB server config
RERANK_THUMBNAIL_EDGE = 768  # long-edge px for rerank thumbnails; None = full-res
GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Rerank is a coarser triage than answering, so it can point at a cheaper/faster
# model. Defaults to GEMINI_MODEL; override with RERANK_MODEL in .env.
RERANK_MODEL = os.getenv("RERANK_MODEL") or GEMINI_MODEL
# LLM-as-judge for the eval harness (eval/run_eval.py --judge). Kept separate from
# GEMINI_MODEL so the judge can differ from the system under test.
EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL") or GEMINI_MODEL

# Reliability knobs for outbound Gemini calls (see src/gemini_client.py).
GEMINI_TIMEOUT_S = float(os.getenv("GEMINI_TIMEOUT_S", "60"))   # per-request timeout
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))  # attempts on transient errors

# Logging (see src/logging_setup.py). LOG_JSON emits one JSON object per line.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_JSON = os.getenv("LOG_JSON", "false").strip().lower() in ("1", "true", "yes")

# HTTP serving (see src/server.py). All default to sane local values, so the server
# runs with no .env edit; override any of them in .env for a real deployment.
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
# Comma-separated CORS origins allowed to call the API from a browser. Defaults to the
# Vite dev server (the ui/ app); set "*" to allow any origin.
CORS_ALLOW_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]
# Reject PDF uploads to POST /ingest larger than this (megabytes).
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# Built Vite bundle (ui/). When present, the server mounts it at "/" so the API and
# the UI are one origin - the deployment shape (see Dockerfile). Absent in a dev
# checkout that has never run `npm run build`, which is why the mount is guarded.
UI_DIST_DIR = ROOT_DIR / "ui" / "dist"

# --- API auth + rate limiting (see src/auth.py, src/ratelimit.py) ---
# Shared secret required in the X-API-Key header on every endpoint except /health and
# the /images static mount. EMPTY DISABLES AUTH - the default, so local dev and the
# test suite need no setup. Set it for anything reachable beyond localhost: /ingest
# spends GPU + Gemini budget and DELETE /corpus/{pdf} destroys data.
API_KEY = os.getenv("API_KEY", "")
# Per-client-IP request caps. The server is single-worker by construction (one warm
# GPU model behind an asyncio.Lock), so an in-process counter is exact, not an
# approximation. 0 disables either limit.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
RATE_LIMIT_INGEST_PER_HOUR = int(os.getenv("RATE_LIMIT_INGEST_PER_HOUR", "10"))
# Read the client IP from X-Forwarded-For instead of the socket peer. Only enable when
# a trusted reverse proxy sets that header - otherwise any client can forge it and get
# a fresh rate-limit bucket per request.
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").strip().lower() in ("1", "true", "yes")


def validate() -> None:
    """Fail fast on missing required configuration.

    Called by the CLIs / server at startup so a misconfiguration surfaces
    immediately with a clear message, instead of an opaque auth error at the
    first Gemini call (GEMINI_API_KEY otherwise defaults to "").
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "key (see README) before running the pipeline."
        )