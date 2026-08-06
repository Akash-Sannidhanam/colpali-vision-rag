"""Central configuration: env vars, model names, paths, and Qdrant settings."""

import math
import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent.parent


def _dir_from_env(var: str, default: Path) -> Path:
    """A directory setting, overridable by env, always resolved to an absolute path.

    Env-overridable so a deployment can point its state at a mounted disk instead of
    bind-mounting over the app directory. Resolved because these are compared against
    each other - `server._to_url` takes a page image's path relative to PAGE_IMAGES_DIR,
    which a relative override would break.
    """
    return Path(os.getenv(var) or default).resolve()


# The three persistent data directories. **They are one backup unit with the Qdrant
# storage**: the vectors in Qdrant and the page PNGs here are halves of one corpus, and
# restoring either without the other leaves an index whose every hit is dropped for a
# missing image (see vector_store._fetch, and index_health for how that is reported).
PDFS_DIR = _dir_from_env("PDFS_DIR", ROOT_DIR / "pdfs")
PAGE_IMAGES_DIR = _dir_from_env("PAGE_IMAGES_DIR", ROOT_DIR / "page_images")
# Cropped/annotated chart slices produced at answer time. Under page_images/ but
# never re-ingested (ingest only globs pdfs/).
CROPS_DIR = PAGE_IMAGES_DIR / "crops"
# Qdrant server URL (e.g. http://localhost:6333). When unset, fall back to the
# embedded on-disk store at QDRANT_PATH - convenient for quick prototyping/tests
# with no container running. Point the app at the Dockerized server by setting
# QDRANT_URL in .env (see docker-compose.yml).
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # None for the local unauthenticated container
QDRANT_PATH = _dir_from_env("QDRANT_PATH", ROOT_DIR / "qdrant_data")  # embedded fallback store

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
# How many visual tokens (patches) the model is allowed to see per page. **The single
# largest lever on ingest speed**, because the ViT's attention is quadratic in patch count,
# so cutting the budget pays superlinearly. Measured on this box (M5, bf16), forward pass
# only:
#
#   budget   patches   forward   pages/min   speedup
#      768       755    6.48 s        9.3      1.00x   <- the checkpoint's own value
#      512       486    3.18 s       18.9      2.04x
#      384       385    2.60 s       23.1      2.49x
#      256       263    1.64 s       36.6      3.95x
#
# **This is not RENDER_DPI, and lowering DPI does not substitute for it.** The processor
# downscales every page to its own pixel budget (`max_pixels`, 602112 = 768*28*28 in this
# checkpoint) long before the model sees it: a 150-DPI page is 1275x1650 = 2.1 MP and
# reaches the model as 672x868. The model already reads pages at ~79 DPI equivalent, so DPI
# buys render time and disk, not embedding time. It still matters for the page PNGs, which
# the answer step and the crops use at full resolution.
#
# Unset means "whatever the checkpoint ships", which is what every stored vector was built
# with - so adding this knob did not invalidate the index. Setting it, even to the value the
# checkpoint already uses, changes EMBED_VERSION and re-embeds: erring toward a needless
# re-ingest rather than toward silently keeping vectors from a different budget.
try:
    EMBED_VISUAL_TOKENS = int(os.environ["EMBED_VISUAL_TOKENS"]) if os.getenv(
        "EMBED_VISUAL_TOKENS") else None
except ValueError as e:
    raise RuntimeError(
        f"EMBED_VISUAL_TOKENS must be an integer, got {os.environ['EMBED_VISUAL_TOKENS']!r}. "
        "It is how many patches the model may see per page; leave it unset to use the "
        "checkpoint's own budget."
    ) from e
# Identifies what produced a stored vector. Written into every point's payload so an
# incremental ingest can tell a still-current page from a stale one: changing the model,
# the render DPI or the visual-token budget invalidates every embedding while leaving the
# PDF bytes untouched, which a content hash alone would miss (see src/ingest.py).
#
# The exact inverse of EMBED_BATCH_SIZE's rationale below: batch size stays *out* because
# batching is vector-identical, and the token budget goes *in* because it changes every
# vector on the page.
EMBED_VERSION = f"{COLPALI_MODEL}@{RENDER_DPI}" + (
    f"@{EMBED_VISUAL_TOKENS}" if EMBED_VISUAL_TOKENS is not None else "")

# Env-overridable for the same reason the retrieval knobs are: an experiment arm should be a
# command-line prefix, not a code edit. Here it is what lets an arm that needs its *own index*
# (anything that changes EMBED_VERSION - a model, a DPI, a visual-token budget) be built and
# evaluated without destroying the pinned one, and re-run later without re-ingesting:
#   EMBED_VISUAL_TOKENS=512 COLLECTION_NAME=pdf_pages_vt512 ... src/ingest.py --rebuild
#   EMBED_VISUAL_TOKENS=512 COLLECTION_NAME=pdf_pages_vt512 ... eval/run_eval.py
# Arms cannot collide: vector_store derives its physical-collection prefix from this name and
# only treats an all-digit suffix as a version, so `pdf_pages`'s sweep skips
# `pdf_pages_vt512_1` and vice versa. See validate() for the one name shape that would break
# that, and _list_physical_versions for the guard itself.
COLLECTION_NAME = os.getenv("COLLECTION_NAME") or "pdf_pages"
VECTOR_DIM = 128 # ColQwen emits one 128-d vector per patch
# Both env-overridable for the same reason RENDER_DPI is: an eval arm should be a
# command-line prefix (`RERANK_K=3 uv run python eval/run_eval.py ...`), not a code
# edit that has to be reverted before the comparison run. load_dotenv() does not
# override an already-set variable, so the prefix wins over .env.
# 12, not 10: two extra slots are the single cheapest lever on cross-document coverage
# measured here - candidate_coverage_avg 0.700 -> 0.800 on their own, with recall@1/@3
# flat and no gold page leaving the slate. MAX_PAGES_PER_DOC takes it the rest of the
# way. See the slate-diversity pass in docs/ENGINEERING_LOG.md.
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "12"))  # candidate pages pulled from Qdrant per query
# 3, not 2: at 2 the reranker spent both slots inside one document on cross-document
# questions, and the answer step filled the missing half from parametric memory - with
# numbers that were plausible and *wrong* (BERT-base "108M" for 110M, BEIR "19 datasets"
# for 18). Measured over 83 questions: 3 genuine answer/citation fixes, zero regressions
# across 144 paired row comparisons, for +7.8% cost and flat latency. See the
# instrument-sharpening pass in docs/ENGINEERING_LOG.md.
RERANK_K = int(os.getenv("RERANK_K", "3"))       # pages kept after the Gemini rerank pass
                                                 # (a cap when RERANK_ADAPTIVE)
# When true, rerank keeps a *variable* number of pages (1..RERANK_K) - only the
# pages the model judged relevant, instead of always topping up to RERANK_K. Trades
# a little predictability for answer precision. Off by default until an eval diff
# proves it wins (docs/ENGINEERING_LOG.md retrieval-quality pass).
RERANK_ADAPTIVE = os.getenv("RERANK_ADAPTIVE", "false").strip().lower() in ("1", "true", "yes")
# Slate diversity: the most pages any one pdf may occupy in the RETRIEVE_K candidate
# slate (0 = no cap, today's behaviour). ColQwen2's MaxSim on a two-part question is
# dominated by whichever document matches more query tokens, and it takes every slot -
# on three baseline rows the top-10 was ten pages of a *single* PDF, shutting the second
# gold document out entirely. That makes candidate_coverage_avg a hard ceiling on
# gold_coverage_avg which no RERANK_K value can lift. See the slate-diversity pass in
# docs/ENGINEERING_LOG.md.
#
# 5, not 4: 4 scores higher coverage (0.850 vs 0.825) but evicts a gold page from the
# slate entirely on every arm it was measured in. colpali-avg-ndcg is a *single*-document
# question whose gold is the 5th colpali.pdf page in the ranking, so any cap below 5 drops
# it and backfills with six documents that have nothing to do with the question. 5 takes
# the full coverage win with zero gold pages lost - the same zero-regressions bar the
# RERANK_K decision was adopted on.
MAX_PAGES_PER_DOC = int(os.getenv("MAX_PAGES_PER_DOC", "5"))
# How much wider than RETRIEVE_K to fetch when the cap is on, so capped-out pages are
# backfilled from deeper in the ranking rather than shrinking the slate. Only read when
# MAX_PAGES_PER_DOC is set; 2.0 is where coverage saturates on this corpus (a 20-deep
# pool scores the same as 25/30/40/50).
CANDIDATE_FANOUT = float(os.getenv("CANDIDATE_FANOUT", "2.0"))
# Query decomposition: split a two-part question into halves, embed each, and fuse the
# rankings (see src/query_decompose.py). MAX_PAGES_PER_DOC caps how much of the slate one
# document may hold, but a cap cannot recover a page retrieval never ranked at all - and
# xdoc-splade-vocab-dpr-dense's second gold document sits outside even the top-50. The
# remaining coverage headroom after the slate-diversity pass is entirely retrieval-side,
# and this is the only lever left for it.
#
# On: the judged run took citation_accuracy 0.9315 -> 0.9589, judge_accuracy 0.9178 ->
# 0.9452 and rerank_recall to 1.0 (every answerable question's gold page now survives
# into the answer step). It costs recall@1/@3 - the halves rank the top of the slate
# worse than the whole question did - but that cost does not reach the answer, because
# RERANK_K=3 picks from a 12-page slate and what matters is gold being *in* it.
#
# The splitter is a heuristic, and every cross-document question in eval/dataset.jsonl
# happens to be phrased "<A>, and <B>?" - so a win on that slice alone would be
# indistinguishable from memorising the dataset. eval/dataset_paraphrase.jsonl is the
# hold-out that tells them apart: it moved coverage 0.7917 -> 0.8333 on deliberately
# varied phrasings, from the 5 of 12 rows the splitter fires on at all.
# See the query-decomposition pass in docs/ENGINEERING_LOG.md.
QUERY_DECOMPOSE = os.getenv("QUERY_DECOMPOSE", "true").strip().lower() in ("1", "true", "yes")
# Most parts a question may be split into, beyond the original. Bounds how wide the
# fused pool can get on a listy question; the whole question is always embedded too, so
# a clause past the cap is no worse off than it is with decomposition off.
MAX_SUBQUERIES = int(os.getenv("MAX_SUBQUERIES", "2"))
# How much the *whole* question's ranking counts for when its halves are fused beside
# it. Not cosmetic: RRF rewards agreement between rankings, so at weight 1.0 the query
# that could not find the second document out-votes the half that could - measured, an
# equal-weight fusion pushed dpr.pdf p3 (rank 4 in its own half's results) out of the
# slate in favour of two dpr.pdf pages the whole question already ranked, and bought
# *zero* coverage for 9 rows of worsened gold rank.
#
# 0, so the whole question is dropped from the fusion entirely and its embed skipped.
# Kept as a knob rather than deleted because it is what makes the rejected equal-weight
# design reproducible: DECOMPOSE_ORIGINAL_WEIGHT=1 restores it exactly.
DECOMPOSE_ORIGINAL_WEIGHT = float(os.getenv("DECOMPOSE_ORIGINAL_WEIGHT", "0.0"))
# Binary quantization is lossy; the fast quantized pass pulls RETRIEVE_K * this many
# candidates, then rescores them against the full-precision vectors on disk. Higher
# recovers recall@1 that quantization costs, at a little more disk I/O per query.
RESCORE_OVERSAMPLING = float(os.getenv("RESCORE_OVERSAMPLING", "2.0"))
UPSERT_BATCH_SIZE = 8    # pages per Qdrant upsert flush; small enough that a batch's
                         # multivector payload (~1.4 MB/page) stays well under Qdrant's
                         # REST size limit, even on the default 32 MB server config
# Pages per ColQwen2 forward pass during ingest. The embed step is 98% of a page's ingest
# cost (measured: bench/reports/ingest_baseline.json - 8.242 s of an 8.413 s page, against
# 0.123 s render and 0.047 s save), so this is the one knob that moves indexing throughput;
# everything else is rounding error.
#
# Deliberately NOT part of EMBED_VERSION. Batching SHOULD be vector-identical when the
# batching implementation correctly trims padding via the attention mask (embedder._embed_batch).
# However, backend-specific bugs (e.g., MPS + bfloat16) can violate this assumption, and
# transformers processor upgrades can silently change embeddings. Verify equivalence with
# `profile_ingest.py --verify-equivalence` after any change to device, dtype, model, torch
# or transformers versions. Folding batch size into the fingerprint would re-embed the whole
# corpus every time someone tuned it; processor/device settings are NOT captured and require
# manual reindexing when changed (see docs/ENGINEERING_LOG.md).
#
# **This value is untuned.** It cannot be measured on an Apple-Silicon box: MPS + bfloat16
# miscomputes a batched forward pass, so embedder._batching_is_supported refuses to batch
# there at all and every arm of the sweep here ran at an effective batch of 1. 4 is therefore
# a middle guess that only ever engages off MPS (in practice CUDA). To actually tune it, run
#   PYTHONPATH=. uv run python scripts/profile_ingest.py --batch-sizes 1,2,4,8,16
# on a CUDA box and read `effective_batch_size` in the report - an arm whose effective size
# is below its requested one is not a measurement of that size. Too large is self-correcting:
# embed_images halves and retries on an out-of-memory error rather than losing a long ingest.
# Env-overridable so a sweep arm is a prefix, like the retrieval knobs.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "4"))
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
# How long a request may wait for the one GPU-resident model before the server sheds it
# as 503 + Retry-After (see src/gpu_arbiter.py). An ingest yields the GPU between page
# batches, so a queued query normally waits about one page - this bound is for a wedged
# server, not for normal contention. 0 waits forever (the pre-arbiter behaviour).
GPU_WAIT_TIMEOUT_S = float(os.getenv("GPU_WAIT_TIMEOUT_S", "60"))

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
    """Fail fast on missing or unusable required configuration.

    Called by the CLIs / server at startup so a misconfiguration surfaces
    immediately with a clear message, instead of an opaque auth error at the
    first Gemini call (GEMINI_API_KEY otherwise defaults to "").
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "key (see README) before running the pipeline."
        )
    # float() happily parses "inf" and "nan", and round() raises on both - which would
    # take out *every* search rather than failing once at startup. Checked here rather
    # than at import so a bad value cannot break tooling that merely imports config.
    if not math.isfinite(CANDIDATE_FANOUT):
        raise RuntimeError(
            f"CANDIDATE_FANOUT must be a finite number, got {CANDIDATE_FANOUT!r}. "
            "It is a multiplier on RETRIEVE_K (2.0 = fetch twice the slate size)."
        )
    # `pdf_pages_7` is the shape vector_store gives a *physical* collection of the alias
    # `pdf_pages`, so an alias by that name would be swept as a stale version of another
    # alias mid-rebuild - silently deleting the arm you were measuring. Cheaper to refuse
    # the name than to make the sweep clever enough to tell them apart.
    if not COLLECTION_NAME or re.search(r"_\d+$", COLLECTION_NAME):
        raise RuntimeError(
            f"COLLECTION_NAME must be non-empty and must not end in _<digits>, got "
            f"{COLLECTION_NAME!r}. That suffix is how vector_store names the physical "
            "collections behind an alias, so such a name collides with another alias's "
            "versions. Use e.g. 'pdf_pages_vt512' rather than 'pdf_pages_512'."
        )
    # A budget below one patch cannot describe a page at all, and the processor would either
    # raise deep inside a forward pass or hand back something empty that indexes cleanly.
    if EMBED_VISUAL_TOKENS is not None and EMBED_VISUAL_TOKENS < 1:
        raise RuntimeError(
            f"EMBED_VISUAL_TOKENS must be at least 1, got {EMBED_VISUAL_TOKENS!r}. "
            "It is how many patches the model may see per page; leave it unset to use the "
            "checkpoint's own budget."
        )
    # Same rationale: a zero or negative batch would make the ingest loop embed nothing
    # and store an empty index, which is far worse than refusing to start.
    if EMBED_BATCH_SIZE < 1:
        raise RuntimeError(
            f"EMBED_BATCH_SIZE must be at least 1, got {EMBED_BATCH_SIZE!r}. "
            "It is how many pages share one ColQwen2 forward pass during ingest."
        )