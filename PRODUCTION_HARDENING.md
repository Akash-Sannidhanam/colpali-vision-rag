# Production-Hardening Roadmap

A staged plan to take this system from a sharp research prototype to something
closer to production grade. The pipeline works well — the gaps are operational:
every query cold-loads the ~2B ColQwen2 model, there's no service surface, only
the two pure helpers are tested, and outside `reranker.py` there's no logging,
error handling, or retries.

The work is a **broad hardening pass** — breadth over depth — hitting the single
highest-leverage improvement in four areas: **reliability, observability,
evaluation, and a warm serving layer + UI**. It's sequenced so a shared
foundation lands first and each later phase builds on it. Every phase is
independently shippable.

Work happens on branch **`production-hardening-pass`**.

## Status at a glance

| Phase | Area | Status |
|-------|------|--------|
| 0 | Shared foundation (config, Gemini client, logging, `run_query` seam) | ✅ **Done** (`e1401af`) |
| 1 | Reliability (route calls through client, graceful answerer, atomic ingest) | ✅ **Done** |
| 2 | Observability (request IDs, node timing, token/cost, tracing) | ✅ **Done** |
| 3 | Warm serving (FastAPI) + Vision RAG UI (React/Vite) | ✅ **Done** |
| 4 | Evaluation (retrieval + answer-quality suite) | ✅ **Done** |
| 5 | Packaging & CI (app Dockerfile, compose service, ruff/mypy/pytest CI) | ✅ **Done** (follow-on) |

**Guiding principle:** one top lever per area, scoped tight. Reuse existing
patterns — the `reranker.py` `try/except → Qdrant top-k` fallback, `_valid_order`,
the lazy module-global singletons (`_model`/`_processor` in `embedder.py`,
`_client` in `vector_store.py`), `image_part` reuse across answer/rerank, and
`close_client()` in a `finally`.

---

## Phase 0 — Shared foundation ✅ DONE

The building blocks the later phases consume. Landed but **not yet wired into the
pipeline** — Phase 1 routes the Gemini calls through the new client, Phase 2 adds
the node logging. Verified via unit tests on the pure logic.

**Shipped:**
- **`src/config.py`** — new knobs `GEMINI_TIMEOUT_S`, `GEMINI_MAX_RETRIES`,
  `RERANK_MODEL` (defaults to `GEMINI_MODEL`), `LOG_LEVEL`, `LOG_JSON`; plus
  `validate()` that fails fast on an empty `GEMINI_API_KEY` (it otherwise defaults
  to `""` and dies opaquely at the first Gemini call).
- **`src/gemini_client.py`** *(new)* — one choke point for all Gemini traffic:
  cached client, per-request timeout, `tenacity` retry/backoff on transient errors
  (429 / 5xx / network only — never auth or 400), and per-call token +
  estimated-cost logging. Returns the raw SDK response, so callers keep their
  existing `.parsed` / `.text` handling.
- **`src/logging_setup.py`** *(new)* — structured stdlib logging: human lines by
  default, one JSON object per line when `LOG_JSON=true`; `extra={...}` fields
  render in both modes.
- **`src/main.py`** — extracted a pure `run_query(question) -> dict` seam (no
  printing / file-opening / client teardown) reused by the CLI, the future server,
  and the eval harness; `run()` is now the CLI wrapper and calls `validate()`.
- **Deps** — added `tenacity`. `fastapi`/`uvicorn[standard]`/`python-multipart`
  and the `streamlit` UI group are deferred to Phase 3.
- **Tests** — `tests/test_gemini_client.py` (retry predicate, token/cost logging),
  `tests/test_logging_setup.py` (formatter). Full suite green (25 passed).

---

## Phase 1 — Reliability ✅ DONE

**Shipped:** all four items landed and were verified with unit tests (`tests/test_answerer.py`,
`tests/test_vector_store.py`, extended `tests/test_reranker.py`; suite green at 47) plus a
live end-to-end pass against the Dockerized Qdrant server — baseline ingest, a query answering
through the alias with token/cost logs, an atomic re-ingest swap (`pdf_pages_1→2→3`, old
collections deleted), a hard-kill (`SIGKILL`) mid-build that left the previous index fully intact
and still answering, and a recovery ingest that swept the orphaned partial. One pre-existing bug
surfaced and was fixed as part of making server ingest reliable: multi-page ColQwen2 multivector
batches (~1.4 MB/page) exceeded Qdrant's default 32 MB REST payload limit, so
`UPSERT_BATCH_SIZE` was lowered to 8 **and** `QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=256` added to
`docker-compose.yml`.

- **Route `answerer.answer` and `reranker.rerank` through `gemini_client.generate`**
  — timeouts + retries for free; drops the per-call `genai.Client()` construction.
  Have `reranker` pass `RERANK_MODEL` (already a config knob).
- **Harden `answerer.py`** — wrap the call + parse in the same `try/except` shape
  `reranker.py` already uses, returning a graceful not-found citation
  `{"answer": "<couldn't read the pages>", "found": False, "source_page": 0, "box": []}`
  so `highlight_node`'s existing guards (`graph.py:42-47`) skip cleanly instead of
  crashing. (Currently a malformed Gemini response raises out of `answer_node`.)
- **Atomic ingest** in `vector_store.py` + `ingest.py` — build into a versioned
  physical collection (`pdf_pages_<n>`) and **alias-swap** `COLLECTION_NAME` onto it
  via `update_collection_aliases`, then delete the old physical collection. A
  mid-ingest failure leaves the previous index serving. `search`/`upsert` already
  reference `COLLECTION_NAME`, which Qdrant resolves through the alias transparently.
  (Keep the embedded on-disk fallback on the simpler `reset=True` path — aliases are
  the server story.) Replaces the current wipe-before-ingest.
- **Qdrant health check** — a `ping()` (`client.get_collections()`) for server
  startup and `/health`, with a clear error if unreachable (today it raises deep in
  `search`).

**Files:** `src/answerer.py`, `src/reranker.py`, `src/vector_store.py`, `src/ingest.py`.
**Verify:** unit-test the hardened `answerer` fallback with a stubbed `gemini_client`
returning garbage → asserts a not-found citation, no raise (this also finally covers
the `answer_node → highlight_node` wiring). Interrupt an ingest mid-run → old index
still answers.

---

## Phase 2 — Observability ✅ DONE

**Shipped:** one query is now legible end to end. A per-query `request_id`
(`contextvar`, bound in `run_query`) is stamped onto **every** log line by a
`logging_setup._RequestIdFilter` on the root handler — so the gemini calls, node
timings, degradation warnings, and the final summary all correlate — essentially for
free, because `request_id` isn't in `_RESERVED` and the existing formatter renders
it. Verified with unit tests on the pure logic (`tests/test_request_context.py`,
`tests/test_graph.py`, `tests/test_main.py`, extended `tests/test_logging_setup.py` /
`tests/test_gemini_client.py`; suite green at 58) plus a live JSON-log query showing a
shared `request_id`, per-node `latency_ms`, and per-call token counts for both
`rerank` and `answer`. Scope grew slightly beyond the original three items to fold in
the cheap adjacent wins the code audit surfaced.

- **Structured logs across the pipeline** — `src/request_context.py` *(new)* holds the
  `request_id` + a token/cost accumulator in `contextvar`s (per-thread/task isolation,
  ready for the Phase 3 server). `graph.py`'s `_timed(name, fn)` wraps each node at
  registration (nodes stay pure, so the direct-call tests are unaffected) to log
  `node start` / `node end` + `latency_ms`. CLI `print()`s untouched.
- **Gemini token/cost accounting** — `gemini_client._log_usage` folds each call's
  tokens/cost into the request accumulator via `record_usage`; `run_query` logs a
  `query complete` summary with total `latency_ms` and aggregated
  tokens / cost / `gemini_calls`.
- **Easy wins (beyond original scope)** — per-call Gemini `latency_ms` + retry
  `attempts` on the `gemini call` line (plus a `before_sleep` WARNING per retry); total
  query latency; and a fix for the previously-silent `reranker.py` fallback — both
  degradation paths now log a `degraded` / `stage`-tagged WARNING carrying the
  `request_id`.
- **LangSmith tracing (opt-in, env only)** — `LANGSMITH_TRACING` / `LANGSMITH_API_KEY`
  documented in `.env.example` + README (`langsmith` is already installed transitively
  via `langgraph`, so no dependency change). `run_query` passes the `request_id` in the
  `graph.invoke` config `metadata`, so traces cross-link to the logs.

**Files:** `src/request_context.py` *(new)*, `src/logging_setup.py`, `src/gemini_client.py`,
`src/graph.py`, `src/main.py`, `src/reranker.py`, `src/answerer.py`, `.env.example`, `README.md`.
**Verify:** `LOG_JSON=true PYTHONPATH=. uv run python src/main.py "…" 2>logs.json` → a
shared `request_id` on every line, per-node `latency_ms`, per-call token counts for
both `rerank` and `answer`, and a `query complete` line with summed totals.

---

## Phase 3 — Warm serving (FastAPI) + Vision RAG UI ✅ DONE

**Shipped:** a warm single-worker FastAPI service plus a React + Vite UI (the user's own
Claude Design "2a" three-column workspace — **Streamlit was dropped**). Verified end to
end: `uvicorn` warms the ~2B model once at boot (`server warm` logged once), two `/query`
calls show no reload; a live query answered "180" for the Q4-revenue chart with a
per-stage token/cost breakdown; static crop/page images serve; CORS allows the Vite
origin; and the browser UI rendered the answer, the CSS bounding-box overlay on the
cited page, the crop slice, the reranked-candidate rail, and the trace disclosure. Full
suite green (74 backend tests + UI typecheck/units).

- **`src/server.py`** *(new)* — FastAPI app. **Lifespan warmup** (`validate` →
  `load_model` → `ping` → `get_graph`) pays the cold start once at boot; shutdown closes
  the Qdrant client. Endpoints: `POST /query` (→ answer + enriched citation + used pages +
  crop/annotated + `meta`, with `?inline=true` for base64 images), `GET /health`
  (model-loaded + `ping`, 503 when down), `GET /corpus` (indexed docs for the rail),
  `POST /ingest` (multipart PDF). One `asyncio.Lock` serializes the GPU model;
  `asyncio.to_thread` keeps the loop free; StaticFiles mounts `page_images/`; CORS to the
  Vite dev origin. Single worker (documented — never `--workers >1`).
- **Per-stage observability** — `request_context` grew a per-stage accumulator, wired via
  `graph._timed`'s `enter_stage`/`exit_stage`; `run_query` folds a `meta` block
  (request_id / latency / usage / `stages[]`) into its return so the HTTP response and the
  future eval harness get it for free. New `get_graph()` compiles the graph once;
  `embedder.is_loaded()`, `vector_store.list_documents()`, and an `ingest.run_ingest()`
  teardown-free seam back the endpoints (the server must **not** reuse `main()`/`run()`,
  which close the shared client).
- **`ui/`** *(new)* — React + Vite + TS. The `2a` workspace: corpus rail (`/corpus` +
  `/health`), conversation with the answer bubble / citation chip / trace disclosure, and
  a document viewer that draws the bounding box as a CSS overlay from `citation.box` over
  the cited page image, with the crop and candidate rail. States: empty / loading /
  results / **not-found** (new — the API produces `found:false`) / error, plus an ingest
  modal. Design tokens ported from the mockup as CSS variables.
- **Deps** — added `fastapi`, `uvicorn[standard]`, `python-multipart`. (No Streamlit.)

**Files:** `src/server.py` *(new)*, `src/main.py`, `src/graph.py`, `src/embedder.py`,
`src/vector_store.py`, `src/ingest.py`, `src/request_context.py`, `src/config.py`,
`pyproject.toml`, `tests/test_server.py` *(new)* + extended context/graph tests, `ui/**`,
`README.md`.
**Deferred** (design outran the backend): multi-region citations, the MaxSim patch
heatmap, normalized confidence %, live-streaming ingest (SSE), and the `4a` animated
walkthrough — layered onto v1 later.

---

## Phase 4 — Evaluation ✅ DONE (the regression guard)

**Shipped:** a labeled dataset + scoring harness that turns "validated on 43 pages"
into a repeatable measurement. A live full+judge run over the 22-question set scored
recall@10 = 1.0, rerank recall = 1.0, citation accuracy = 1.0, substring = 1.0, and
judge = 1.0 (avg 5/5), with recall@1 = 0.77 leaving the reranker real work to do.
Building it also surfaced three under-labeled gold rows (a fact restated on a page the
dataset hadn't listed) — caught by the new per-row `cited` field and fixed by widening
the gold lists, which is exactly the kind of drift the harness exists to catch. Full
suite green (81 backend + the two new eval test files).

- **`eval/dataset.jsonl`** *(new)* — 22 questions over the shipped corpus
  (`attention.pdf`, `colpali.pdf`, `sales_report.pdf`): each row an `id` + question +
  a **list** of gold `{pdf, page}` (a fact can legitimately live on more than one
  page), optional `answer_contains` substrings, and `tags`
  (`chart`/`table`/`figure`/`formula`/`text`) for per-modality slices.
- **`eval/scoring.py`** *(new)* — pure, unit-tested scoring: `load_dataset`
  (jsonl validation naming the bad line), `gold_rank`, `citation_correct` (resolves
  `source_page` against the **reranked** list), `substring_match` (case-insensitive
  any-of, `None` = N/A), `aggregate` (rates over applicable rows only + per-tag), and
  `format_table`. No `src.` imports, no I/O.
- **`eval/run_eval.py`** *(new)* — the CLI. `--retrieval-only` embeds + searches per
  question (no Gemini, runs with no `GEMINI_API_KEY`); full mode reuses the
  `main.run_query` seam, scoring recall@k over the new pre-rerank `candidates`, rerank
  recall, citation correctness, and substring match, with per-row latency/token/cost
  for free from `meta`. `--judge` adds flag-gated LLM-as-judge scoring routed through
  `gemini_client.generate` (new `EVAL_JUDGE_MODEL` knob, `RERANK_MODEL` pattern; a
  judge outage degrades to N/A, never fails the run). A corpus preflight fails fast
  (exit 2) if a gold pdf/page isn't indexed; `--fail-under-recall` is a CI gate
  (exit 1). Writes a JSON report with a `config` snapshot so before/after runs diff
  cleanly.
- **Pipeline seam** — `retrieve_node` now also writes the untrimmed top-k to a new
  `candidates` `RAGState` key (rerank overwrites `retrieved`), so recall@k reflects the
  retrieval the pipeline actually used. A new `tests/test_pipeline_integration.py`
  locks the full compiled-graph flow (rerank-order alignment, fallback, degradation).

**Files:** `eval/dataset.jsonl` *(new)*, `eval/scoring.py` *(new)*, `eval/run_eval.py`
*(new)*, `eval/__init__.py` *(new)*, `src/graph.py`, `src/config.py`,
`tests/test_pipeline_integration.py` *(new)*, `tests/test_eval_scoring.py` *(new)*,
`tests/test_run_eval.py` *(new)*, `README.md`, `.gitignore`.
**Verify:** `GEMINI_API_KEY= PYTHONPATH=. uv run python eval/run_eval.py
--retrieval-only` → recall@k table + report, no key needed; `--judge` → all four
metric families + `purpose=judge` token-logged calls; a mislabeled gold → exit 2;
`--fail-under-recall` breach → exit 1.

---

## Phase 5 — Packaging & CI ✅ DONE (the follow-on)

Originally listed under **Out of scope** below; picked up once the warm server (Phase
3) made a deployable image the obvious next step. Done after the main pass on its own
branches (PRs #12–#14), not `production-hardening-pass`. All checks green on `main`:
ruff clean, mypy clean (17 files), 118 tests, container smoke test passing.

**Shipped:**
- **`ruff` + `mypy` tooling** — a `lint` dependency group in `pyproject.toml`
  (isolated from the ML runtime so CI's lint job installs only these), `[tool.ruff]`
  (default rules + import sorting) and `[tool.mypy]` (configured for the
  namespace-package `src/`, `ignore_missing_imports` for the stub-less ML/vector deps,
  non-strict). Cleared the baseline: import blocks sorted and **7 genuine mypy
  findings** fixed, all behavior-preserving (PIL `Image`/`ImageFile` reassignments,
  `dict | None` payload unpack, tenacity `.statistics` via `getattr`, two documented
  `type: ignore`s).
- **`Dockerfile`** *(new)* + **`.dockerignore`** *(new)* — multi-stage `uv` build
  serving the FastAPI backend (the UI ships separately). Slim runtime + `poppler-utils`,
  non-root `appuser`, `PYTHONPATH=/app`, binds `0.0.0.0:8000`, `/health` HEALTHCHECK
  with a generous start-period for the first-boot model download. On Linux `uv` pulls
  the CUDA 12.8 torch wheels, so the image is GPU-capable with `--gpus all` and
  **auto-falls back to CPU**. `COPY --chown` (not a trailing `chown -R /app`) keeps the
  image at **11 GB** instead of 21.7 GB — the recursive chown would re-copy the
  multi-GB torch venv into a new layer.
- **`docker-compose.yml`** — new `app` service wired to the `qdrant` service
  (`QDRANT_URL` over the compose network, `GEMINI_API_KEY` passthrough, an HF-cache
  volume, a commented `deploy.resources` GPU-reservation block for NVIDIA hosts), so
  `docker compose up` runs the whole stack.
- **`.github/workflows/ci.yml`** *(new)* — on push to `main` + PRs, least-privilege
  `GITHUB_TOKEN` (`contents: read`): a fast `lint` job (ruff, no ML install) and a
  `test` job that installs the full stack and runs `mypy` + `pytest`. mypy lives in the
  test job so it type-checks against real pydantic/PIL/fastapi types rather than `Any`.
- **`vector_store.search()` hit filtering** — drops points whose payload is missing/
  wrong-typed (`pdf`/`page_number`/`image_path`) or whose page image is gone from disk
  (a persisted index outliving a wiped `page_images/`), logging each drop at WARNING so
  a stale index stays visible. Downstream can now assume every hit resolves to a page.

**Files:** `pyproject.toml`, `Dockerfile` *(new)*, `.dockerignore` *(new)*,
`docker-compose.yml`, `.github/workflows/ci.yml` *(new)*, `src/vector_store.py`,
`src/server.py`, `tests/test_vector_store.py`, `README.md`, plus the ruff/mypy baseline
fixes across `src/`.
**Verify:** `uv run ruff check .` + `uv run mypy src eval` clean; `uv run pytest` 118
green; `docker build .` succeeds and a container smoke test imports `src.server`;
`docker compose config` validates.

---

## Out of scope (natural follow-ons, not in this pass)

Security / input validation (PDF size/page caps, Qdrant auth/TLS, query length
limits) and scaling/perf (batch the embedder — it embeds one page at a time today —
query-result cache, incremental content-hash ingest). *(Packaging & CI graduated out
of this list — see Phase 5 above.)*
