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
walkthrough — layered onto v1 later. **Since shipped:** multi-region citations, normalized
confidence %, and SSE ingest (all on `main`), plus the **MaxSim patch heatmap** — a
`POST /heatmap` endpoint (`src/server.py`), backed by a `src/heatmap.py` compute helper, returns a
per-patch query→page similarity grid that the Viewer's "why this page?" toggle paints over the cited
page (`ui/src/components/Viewer.tsx`). Still open: the `4a` animated walkthrough.

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
query-result cache). *(Packaging & CI graduated out of this list — see Phase 5 above.
**Incremental content-hash ingest** graduated too — see the corpus-lifecycle pass
below.)*

---

## Corpus-lifecycle pass (follow-on) ✅ DONE

Work on branch **`feat/corpus-lifecycle`**. The last big *functional* hole once the
UI made ingest interactive: `POST /ingest` passed every PDF in `PDFS_DIR` to a
fresh-collection rebuild, so adding one document re-rendered and re-embedded the whole
corpus through the ~2B model — and nothing could be removed from the index at all.

**Shipped:**
- **Incremental sync is the default** (`ingest.run_ingest`, `vector_store.live_collection`)
  — upserts into the live collection and embeds only documents whose fingerprint moved.
  The fingerprint is `content_hash` (sha256 of the PDF bytes) **plus** `embed_version`
  (`COLPALI_MODEL@RENDER_DPI`), both stored per point: a content hash alone would
  silently keep stale vectors after a DPI or model change. A changed document is deleted
  before re-embedding, since stable `uuid5(pdf, page)` ids would otherwise strand the
  tail pages of a longer previous revision. Sync never prunes — removal is always explicit.
- **Atomicity preserved where it matters** — the incremental path never touches existing
  points, so an interrupted add leaves every other document serving; `--rebuild` keeps the
  original alias-swap path for a genuine wipe.
- **`DELETE /corpus/{pdf}`** — drops vectors (a filter on the now-indexed `pdf` payload
  field, supported in both Qdrant modes), page images, crops, and the source PDF. Takes no
  GPU lock. The path parameter is normalized with `Path(pdf).name` and must be present in
  the index before anything is unlinked; file matching uses anchored regexes rather than
  globs (`pdf_render.page_images_for` / `crop_images_for`) so a document named
  `report_page_1.pdf` can't be caught in `report.pdf`'s sweep.
- **UI** — a hover-revealed remove action with an inline confirm in the corpus rail, and a
  `skip` SSE phase surfaced as "already indexed — unchanged" in the ingest modal.

**Files:** `src/ingest.py`, `src/vector_store.py`, `src/pdf_render.py`, `src/server.py`,
`src/config.py`, `tests/test_pdf_render.py` *(new)* + extended ingest/vector_store/server
tests, `ui/src/{api,types,App}.ts(x)`, `ui/src/components/{CorpusRail,IngestModal}.tsx`,
`ui/src/theme.css`, `README.md`, `CLAUDE.md`, `ENHANCEMENTS.md`.
**Migration:** points indexed before this change carry no fingerprint, so the first sync
after upgrading re-embeds everything once. Run `src/ingest.py --rebuild` once instead.

---

## Retrieval-quality pass (follow-on)

Work on branch **`retrieval-quality-sweep`**. Goal: lift retrieval recall@1 and turn
the eval into a sensitive regression instrument. The eval was **saturated** — every
downstream metric pinned at 1.0 on 22 questions — so no retrieval change could be
proven either way.

**Foundation (shipped):**
- Eval set grown **22 → 53** questions (`eval/dataset.jsonl`): single-page-gold,
  visually-confusable "hard" table/figure lookups mined from the page images.
  De-saturates retrieval — recall@1 now **0.83** with real per-tag headroom (text 0.74).
- Tunable knobs, all defaulting to prior behavior: `RESCORE_OVERSAMPLING` (binary-quant
  rescore depth, `vector_store.search`), `RERANK_ADAPTIVE` (variable rerank count via
  `_valid_order(top_up=…)`), env-overridable `RENDER_DPI` and `COLPALI_MODEL` (the
  embedder auto-selects the `ColQwen2_5` loader for colqwen2.5 checkpoints).
  `run_eval.py` gains a `--fail-metric` gate (gate on a metric with headroom, not the
  saturated recall@10) and per-row top-1 miss diagnostics.

**Lever sweep — every candidate measured, none adopted:**

| Lever | Result | Decision |
|-------|--------|----------|
| Rescore oversampling (1→4×) | no-op — binary quantization is lossless here (server == exact recall) | keep 2.0 |
| Adaptive rerank | no precision gain (citation stays 1.0), small judge dip, ~5% faster | keep off |
| Render DPI 150→220 | +3.8 pts recall@1 / −1.9 pts recall@3, costlier ingest+query, no end-to-end gain | keep 150 |
| Model → colqwen2.5-v0.2 | MPS OOM on the dev box (Apple M5) — needs a bigger GPU | not evaluable locally |

**Takeaway:** on this 43-page corpus the pipeline is already saturated **end-to-end**
(recall@10 = 1.0, rerank_recall = 1.0, citation_accuracy = 1.0) — the reranker recovers
every recall@1 miss, so raw-recall tuning has no downstream effect. The recall@1 gap is
genuine ColQwen2 ranking, not quantization loss. Real gains would need a larger/harder
corpus (distractor docs, which stress `RERANK_K`) or a bigger GPU for the colqwen2.5
upgrade — both deliberately out of this pass's scope.

---

## Eval de-saturation ✅ DONE (the guard can now fail)

**The problem.** Phase 4 shipped a regression guard that could not detect a regression.
`recall@10`, `rerank_recall`, `citation_accuracy`, `substring_accuracy` and
`judge_accuracy` all scored exactly 1.0 — not because the pipeline was perfect but
because `RETRIEVE_K=10` over a 43-page corpus returned 23% of the index per query, so
the gold page could not fail to be in the candidate set and every downstream stage
inherited the ceiling. `--fail-under-recall` existed with nothing to guard. Two gaps
compounded it: `load_dataset` structurally forbade an unanswerable question (`gold` had
to be non-empty), so the hallucination rate was unmeasurable; and all 10 multi-gold rows
lived inside one PDF, so `RERANK_K=2` was never asked to span documents.

**The prediction this pass tested — and half-falsified.** The retrieval-quality pass
concluded that "real gains would need a larger/harder corpus (distractor docs, which
stress `RERANK_K`)". Half of that was wrong. 16 papers / 320 pages of deliberately
confusable material (ColBERT and ColBERTv2 with the same late-interaction prose, BEIR
with the same nDCG tables, PaliGemma which *is* ColPali's base model) took the index
from 43 → 363 pages, and on the **identical 53 questions** moved recall@1, recall@3 and
recall@10 by **exactly zero**. `recall@10` and `rerank_recall` are still 1.0. ColQwen2's
ranking is far more robust to a bigger haystack than the corpus-size hypothesis assumed.

What *did* de-saturate the eval was the **question types**, at ~15 minutes of labeling
against ~3 hours of fetching and ingesting. Recorded here because the cheap lever was
not the one the plan led with.

**The finding that justifies the pass.** `gold_doc_coverage` turns out not to measure
retrieval — it catches **answers that are correct but ungrounded in the retrieved
pages**. On `xdoc-ndcg-cutoffs` the pipeline answered *"ColPali reports nDCG@5… BEIR is
scored using nDCG@10"*, scored correct on citation, substring **and** the judge, while
`beir.pdf` was never in the reranked set: the BEIR half came from parametric knowledge,
not a page it was shown. For a system built on "here is the box where I read this,"
that is the failure that matters, and nothing in the old eval could see it. Under the
same `RERANK_K` starvation the pipeline sometimes does the right thing instead —
`xdoc-colbert-vs-colpali-dim` explicitly declined on the half it couldn't see — and
`gold_coverage_avg` is the only metric that distinguishes the two.

- **`eval/scoring.py`** — `unanswerable: true` rows (explicit flag, not inferred from an
  empty `gold`, so a row that lost its label stays a validation error);
  `abstention_correct`; `gold_doc_coverage` (N/A unless gold spans 2+ PDFs, so
  cross-document questions need no schema flag); confidence calibration
  (`confidence_separation` plus citation accuracy bucketed by the model's self-report);
  `substring_match` normalizes thousands separators.
- **`eval/run_eval.py`** — `run_full` gives unanswerable rows a deliberately different
  row shape (abstention only, judge skipped) so a correct refusal can never read as a
  retrieval miss; `run_retrieval_only` keeps the row but omits `gold_rank`. `--gate
  METRIC:MIN` is repeatable and reports every breach, since one `--judge` run costs ~25
  minutes of GPU.
- **`eval/corpus_manifest.json` + `scripts/fetch_eval_corpus.py`** *(new)* — 16 papers
  pinned by sha256, fetched not committed, failing loudly on a hash mismatch because a
  silently-revised paper would invalidate every stored report.
- **`eval/dataset.jsonl`** — 53 → 69 questions.

**Measurement errors caught and fixed along the way** (4 of the first run's 6 non-1.0
data points were the instrument, not the pipeline): three under-labeled gold lists
surfaced by the per-row `top1`/`cited` fields, a reference that listed only half a
two-part answer so the judge rejected a correct one, and `37,000` scored as a miss
against the label `37000`. Deliberately *not* fixed: the judge rejecting `$150,000` for
a chart labeled "Thousands" showing 150 — that is the judge's noise floor, and tuning it
away would flatter the number rather than measure it.

**Baseline:** `eval/reports/baseline_desaturated.json` (committed; the rest of
`eval/reports/` stays gitignored). recall@1 0.7627, recall@3 0.9322, recall@10 1.0,
rerank_recall 1.0, citation_accuracy 1.0, substring 1.0 (over 58 applicable
rows), judge 0.9831,
**gold_coverage_avg 0.75**, **abstention_accuracy 1.0** (10/10, zero hallucinations),
`confidence_separation` is n/a in the pinned run: it needs at least one wrong citation
to exist and there were none, which is a real flaw in how it is defined - the metric
disappears when the pipeline does well. Across four valid runs of identical code,
citation_accuracy was 0.9661 three times and 1.0 once, gold_coverage_avg 0.6667 three
times and 0.75 once, and `sales-q2-revenue` flipped the judge miss/pass/miss/pass. One
question is worth 0.017 / 0.083 on those denominators, so none is gated tightly.
**Verify:** `uv run python scripts/fetch_eval_corpus.py --verify` → exit 0;
`PYTHONPATH=. uv run python eval/run_eval.py --judge --gate recall@1:0.70 --gate
recall@3:0.88 --gate citation_accuracy:0.91 --gate gold_coverage_avg:0.55 --gate
abstention_accuracy:0.90` → exit 0 at baseline, exit 1 on any breach.

**Still open, deliberately:** `recall@10` and `rerank_recall` remain 1.0 and are not
gated — on this pipeline the retrieval stage genuinely is not the bottleneck at k=10,
and pretending otherwise would guard nothing. The obvious follow-up is raising
`RERANK_K` for multi-document questions, which `gold_coverage_avg` now makes measurable.

## Instrument-sharpening pass (follow-on) ✅ DONE

**The problem.** The de-saturation pass left exactly one metric with both headroom and a
real failure mode behind it — `gold_coverage_avg` — and it could not adjudicate the very
decision it was built for. Over 6 cross-document rows one question moves it 0.083, and
runs of *identical* code had already produced 0.667 and 0.75. A `RERANK_K=2 → 3` result
would have landed inside its own noise floor. Separately, a run against depleted quota
still wrote a pinnable report, and scored `abstention_accuracy` **1.0** while doing it.

- **Degraded-run guard** — `request_context.record_degraded()` counts every call that
  fell through to graceful degradation, called from the `except` branches in
  `reranker.rerank` and `answerer.answer`. It rides to the eval on `meta.degraded_calls`
  for free, because `run_query`'s meta already spreads the usage accumulator.
  `run_eval.degradation_summary` adds judge N/As (a `None` verdict *is* a failed call);
  past `--max-degraded-frac` (default 0.02) the run stamps `degraded_run`, writes
  `degraded_<utc>.json`, skips the gates, and exits 2. Clean runs still record zeros —
  a report that can't say how much of the run reached the model is not much better than
  one written before the guard existed.
- **`eval/diff_reports.py`** *(new)* — pairs two reports by question id: flipped rows,
  improved-vs-regressed counts, averages over the same question set both sides, and the
  config knobs that differ. Refuses to quietly compare across drifted datasets, unknown
  metrics, or a degraded side.
- **`eval/dataset.jsonl`** — 69 → 83 questions; the cross-document slice 6 → 20, over
  document pairs the original six never touched. No new PDFs and no re-ingest: the
  distractor corpus already held these papers as noise, so this makes them load-bearing
  for free. One question is now worth 0.05 rather than 0.083, and the comparison is
  paired, which is what actually buys the power.
- **`src/config.py`** — `RETRIEVE_K` and `RERANK_K` read from env like the knobs beside
  them, so an arm is a command-line prefix rather than an edit to revert.

**Caught by spot-checking three new rows before committing to a full run:** the model
answers "50K questions" where the DocVQA page reads "50,000", so that label was
known-bad on arrival. The same three queries surfaced a real cross-document miss — the
reranker took `donut.pdf` p.8 over any DocVQA page and the answer step supplied the
DocVQA half from parametric knowledge anyway.

**The result: five metric families came off 1.0 at once.** The de-saturation pass had
moved recall@1 and `gold_coverage_avg` and nothing else; every other family was still
pinned at exactly 1.0 and could not report a regression. On the 83-question set:

| metric | 69q baseline | 83q baseline |
| --- | --- | --- |
| recall@10 | 1.0 | **0.9863** |
| rerank_recall | 1.0 | **0.9726** |
| citation_accuracy | 1.0 | **0.9178** |
| substring_accuracy | 1.0 | **0.9014** |
| judge_accuracy | 0.9831 | 0.9041 |
| recall@1 / recall@3 | 0.7627 / 0.9322 | 0.726 / 0.8767 |
| gold_coverage_avg | 0.75 (n=6) | 0.625 (n=20) |
| confidence_separation | n/a | **0.0261** |

`confidence_separation` computes for the first time — it needs at least one wrong
citation to exist, and now some do. At 0.0261 it confirms what the de-saturation pass
suspected: the retrieval-confidence signal separates correct from wrong citations by
almost nothing.

**The finding that decides the RERANK_K question.** All **11** failing rows are
cross-document, every one at `gold_doc_coverage` 0.5 or 0.0. Sharper than the "correct
but ungrounded" case the last pass found: when the second document is not reranked in,
the model does not decline — it fills the gap from parametric memory and gets the number
*slightly wrong*. BERT-base as "108M" (the page says 110M), BERT-large as "334M" (340M),
BEIR as "19 datasets" (18). A plausible wrong number sourced from memory is exactly the
failure a visual-citation system exists to prevent, and nothing before this pass could
see it.

**Baseline for this section:** `eval/reports/baseline_sharpened.json` (83 questions at
`RERANK_K=2`), `degradation` all zeros. Superseded as the *current* baseline by the
`RERANK_K` decision below, but kept: it is arm A of that experiment and the numbers above
are its numbers. The old documented gate of `recall@3:0.88` is retired — the value is now
0.8767, inside its own noise floor.

### The RERANK_K decision ✅ ADOPTED (k=2 → 3)

Three arms over the 83 questions, judge off for B and C (coverage, citation and substring
do not need it, and leaving it out keeps judge variance out of the read). All three ran
with `degradation` at zero.

| | A — k=2 | B — k=3 fixed | C — k=3 adaptive |
| --- | --- | --- | --- |
| `gold_coverage_avg` | 0.625 | 0.625 · **0 flips** | 0.65 · +1, −0 |
| substring | 0.9014 | 0.9577 · +4, −0 | 0.9437 · +3, −0 |
| citation | 0.9178 | 0.9315 · +1, −0 | 0.9178 · 0 flips |
| rerank_recall | 0.9726 | 0.9726 | 0.9863 |
| cost / 83 q | $0.3740 | $0.4032 (+7.8%) | $0.3930 (+5.1%) |
| median latency | 13444 ms | 13740 ms (+2.2%) | 14309 ms (+6.4%) |

**Zero regressions in all six paired comparisons** — 0 of 144 row comparisons moved
backwards. Run-to-run variance produces flips in both directions, so the direction is
real even from one run per arm. `RERANK_K` defaults to 3.

**The pre-registered criterion did not fire, and that is the more useful finding.**
`gold_coverage_avg` — the metric this whole pass existed to make decisive — moved by
nothing on B. The reason is an instrument gap, not an absent effect: **page-level gold
lists are narrower than the set of pages that state each fact.** Arm B fixed the DocVQA
row by reading `docvqa.pdf` p.3 ("The DocVQA comprises 50,000 questions"), which is not
in that row's gold; `beir.pdf` states its 18 datasets on p.7 and p.9 as well as the
labelled p.1–3. Coverage scored both wins as nothing. The pinned `gold_coverage_avg` is
therefore a **lower bound**, and closing the gap is filed in `ENHANCEMENTS.md`.

**Auditing the four substring flips leaves two clean wins, not four** — the audit rule
earning its keep again:

- `xdoc-docvqa-scale-layoutlm-pretraining` — real: "6 million" → "11 million".
- `xdoc-e5-pairs-beir-datasets` — real: read 18 off a BEIR page instead of answering
  "19" from memory.
- `xdoc-vit-base-vs-bert-base-params` — **a hedge, not a fix**: "either 108M (Page 2) or
  110M (Page 3)". Substring passes on "110M" while the answer got vaguer.
- `xdoc-colbert-v1-v2-compression` — **phrasing luck**: substantively the same answer,
  which happened to write "residual compression" this time and match a strict label.

So the genuine gain is two answer fixes plus one citation fix
(`xdoc-rag-generator-dpr-dim`, `None` → `rag.pdf` p.3) for +7.8% cost — a weaker case
than the summary row implies, and adopted on substring/citation rather than the metric
that was pre-registered. Recorded that way deliberately.

**Arm C rejected, narrowly.** `RERANK_ADAPTIVE=true` with `RERANK_K=3` costs no new code
(`_valid_order(top_up=False)` already keeps 1..k), was cheaper, and was the only arm to
move coverage or `rerank_recall`. It gave up one substring win and the citation fix, and
its median latency was *worse* than fixed k=3 — the cost saving was 2.7 points. Kept as
a knob for a corpus where the third page distracts more than it helps.

**Re-pinned at k=3**, an independent `--judge` run that reproduced the arm and moved a
little further: citation 0.9178 → **0.9452**, substring 0.9014 → **0.9577**, judge 0.9041
→ **0.9315**, `gold_coverage_avg` 0.625 → **0.65**, and average latency *down* 15859 →
14786 ms. Retrieval metrics are byte-identical, as they must be — `RERANK_K` is applied
after retrieval. Paired against the k=2 baseline: **0 regressions**, 4 substring and 2
citation improvements. `xdoc-docvqa-scale-layoutlm-pretraining` is a clean sweep, fixing
coverage (0.5 → 1.0), citation and substring at once. The gap between this run's citation
figure and arm B's (0.9452 vs 0.9315) on identical config is the documented run-to-run
variance — both beat k=2, which is the claim.

**Baseline:** `eval/reports/baseline_rerank_k3.json`, `degradation` all zeros. *(Superseded
as the current baseline by the cross-document attribution pass below; kept because this
section cites its numbers.)*
**Verify:** `PYTHONPATH=. uv run python eval/run_eval.py --judge --gate recall@1:0.68
--gate recall@3:0.83 --gate recall@10:0.95 --gate rerank_recall:0.93 --gate
citation_accuracy:0.89 --gate substring_accuracy:0.90 --gate abstention_accuracy:0.90
--gate gold_coverage_avg:0.55 --gate judge_accuracy:0.87` → exit 0 at baseline, exit 1 on
a breach, exit 2 if the run itself degraded. Each threshold leaves roughly three questions
of slack: one question is worth 0.0137 on the recall denominators, 0.0141 on citation and
substring, and 0.05 on coverage.

---

## Cross-document attribution pass (follow-on) ✅ DONE

**The problem.** Every imperfect row in the pinned baseline was a cross-document question
— 12 of 12 — and 10 of them scored `gold_doc_coverage` 0.5. But nothing in a stored report
said *which stage* lost the second document, because coverage scores the reranked set and
`gold_rank` records only the best gold page's rank. The two plausible fixes live in
different stages, so there was no way to scope one. Compounding it, `gold_coverage_avg` was
documented as a lower bound: page-level gold lists name fewer pages than state each fact.

**Shipped:**
- **`candidate_doc_coverage`** — `scoring.gold_doc_coverage` was already stage-agnostic in
  its hits argument, so the retrieval-stage twin is the same function applied to the
  untrimmed Qdrant candidates, not new scoring logic. Read as a pair: candidates 1.0 with
  coverage <1.0 means rerank dropped a page it was offered; equal coverage with both <1.0
  means retrieval never offered it and rerank was blameless; any decrease from candidate to
  reranked coverage is attributed to reranking (potentially alongside retrieval, not
  retrieval alone). `candidate_coverage_avg` joins the summary; `cand_cov` joins the table
  beside `cov`. Scored in `--retrieval-only` too, which needs **no API key and no Gemini
  spend**.
- **`candidate_pages` / `reranked_pages` per row** — `gold_rank`, `rerank_hit` and both
  coverages are pure functions of (pages, gold), so a label change becomes re-scorable
  **offline** instead of costing a judged re-run. It paid for itself inside this pass: the
  sweep's effect on every retrieval metric was computed from the stored report before the
  re-run, and the re-run then reproduced those numbers exactly (`candidate_coverage_avg`
  0.700 predicted, 0.700 measured; recall@1 0.7397 predicted, 0.7397 measured).
- **Gold-label sweep** — all 20 cross-document rows, 11 gained pages, plus the stronger
  `["450%", "6 identical"]` that `5b688e5` deferred here.

**Finding 1: the gold-label gap was mostly a myth.** The sweep moved coverage on **one
row**. The pinned figure was a lower bound by 0.025, not by the wide margin the note in
`ENHANCEMENTS.md` feared. `docvqa.pdf` p.3 ("The DocVQA comprises 50, 000 questions") is
real and now labelled; the rest of the suspected gap is not there. Recorded because the
instrument worry that motivated the sweep was largely unfounded, which is worth knowing
before trusting the next such worry. The sweep also found a trap: `pdftotext` matches
**running headers**, and "OCR-free Document Understanding Transformer" is `donut.pdf`'s
title on every odd page. Counting page furniture would have made half that document gold
and turned document-level coverage into a tautology — only prose that states the fact counts.

**Finding 2: retrieval is the ceiling, not rerank — and `RERANK_K` never could have been.**
Scored against the three-case rule above, the 12 misses split **11 retrieval-only, 1
rerank-only, 0 mixed**. The single rerank-only case is
`xdoc-donut-ocr-free-docvqa-questions`, where retrieval offered both documents but
reranking dropped one; on the other 11 rows retrieval never offered both. (That no row is
the mixed case is why the two-case reading this section originally shipped with reached
the right answer despite being wrong in general; the third case is now covered by a test
rather than by luck.) So `candidate_coverage_avg` is a hard ceiling on
`gold_coverage_avg`, and at 0.675 vs 0.700 coverage already sits at **96% of what
retrieval offers**. This retroactively explains why the `RERANK_K` decision's
pre-registered metric did not fire: no value of `RERANK_K` could have moved it.

**Finding 3: the mechanism is monopolisation, not ranking.** Seven misses are the second
document being *entirely absent* from the top-10, and on three rows the top-10 is **ten
pages of a single PDF** — `colpali.pdf` ×10 shuts out `paligemma.pdf`, `albert.pdf` ×10
shuts out `bert.pdf`, `rag.pdf` ×10 shuts out `dpr.pdf` — with a fourth at 9 of 10.
ColQwen2's MaxSim on a two-part query is dominated by whichever document matches more
query tokens, and it takes every slot.

**The probe that scopes the fix** (free, deterministic, `RETRIEVE_K=50 --retrieval-only`).
Two results, and the first is the one that settles the question:

- **`candidate_coverage_avg` is 0.975 at k=50, against 0.700 at k=10.** Retrieval reaches
  both gold documents on 19.5 of 20 cross-document rows once the slate is deep enough. The
  ceiling is therefore **not ColQwen2's ranking ability** — the second document is ranked,
  it is crowded out of a 10-slot slate. That rules out "the model cannot find it" and
  leaves candidate *diversity* as the entire problem.
- For **6 of the 7** shut-out documents the gold page sits at global rank 11–41 but at rank
  **1–4 among its own document's pages** — beir 11/#1, paligemma 15/#1, siglip 15/#2, bert
  16/#3, dpr 20/#4, dpr 41/#4.

So a **per-document cap of 4** on the 10-slot slate recovers six of seven without deepening
retrieval at all. Only `xdoc-splade-vocab-dpr-dense`'s `dpr.pdf` page is outside the top-50
entirely and would need query decomposition. That is the next pass, and it is now scoped on
measurement rather than on a hypothesis.

**Audit of the flipped rows** (the standing rule earning its keep):
- `gold_rank` improved on 2, regressed on 0 — exactly the two rows the offline re-score
  predicted. `gold_doc_coverage` improved on 1, regressed on 0.
- `xdoc-e5-pairs-beir-datasets` lost substring **and** the judge — the model answered "19
  information retrieval datasets" where it had said 18. Its labels did not change and
  coverage is still 0.5, so this is **the parametric-memory failure recurring**, not an
  instrument artifact. The row is unstable precisely because it is ungrounded.
- `xdoc-rag-generator-dpr-dim` lost its citation by *declining to cite at all* rather than
  by citing wrongly — arguably better behaviour, scored as a regression.
- `attn-posenc-geometric-progression` lost the judge because the answer step emitted
  **Zalgo text** — `10000 · 2` followed by a combining-character dump. Rare model
  degeneration, and worth noting that `substring_match` scored it **True** on the intact
  prefix while the judge caught it. The judge is the backstop for garbage output.
- `sales-q2-revenue` flipped to pass: the documented noise-floor row.

**Baseline:** `eval/reports/baseline_swept.json` (83 questions, `RERANK_K=3`),
`degradation` all zeros.

| metric | k=3 baseline | this baseline |
| --- | --- | --- |
| recall@1 / recall@3 | 0.726 / 0.8767 | 0.7397 / 0.9041 |
| recall@10 / rerank_recall | 0.9863 / 0.9726 | 0.9863 / 0.9863 |
| citation_accuracy | 0.9452 | 0.9452 |
| substring_accuracy | 0.9577 | 0.9444 |
| judge_accuracy | 0.9315 | 0.9178 |
| gold_coverage_avg | 0.65 | 0.675 |
| **candidate_coverage_avg** | n/a | **0.70** *(the ceiling)* |
| confidence_separation | 0.0247 | 0.0226 |

Retrieval metrics moved because the sweep added gold pages; the answer-side metrics moved
within the documented run-to-run variance, audited row by row above.

**Verify:** `PYTHONPATH=. uv run python eval/run_eval.py --judge --gate recall@1:0.68
--gate recall@3:0.86 --gate recall@10:0.94 --gate rerank_recall:0.94 --gate
citation_accuracy:0.90 --gate substring_accuracy:0.90 --gate abstention_accuracy:0.90
--gate gold_coverage_avg:0.52 --gate candidate_coverage_avg:0.65 --gate
judge_accuracy:0.87` → exit 0 at baseline. Most thresholds leave ~3 questions of slack
(one question is worth 0.0137 on the answerable denominators, 0.05 on the two coverages,
0.1 on abstention). **`candidate_coverage_avg` is deliberately the tightest gate at one
question of slack**: it is pure retrieval, measured identical across three runs here, so it
needs no allowance for LLM variance — and it is the metric the next pass exists to move.

---

## Slate-diversity pass (follow-on) ✅ DONE

**The problem.** The attribution pass left one lever: `candidate_coverage_avg` (0.700) is a
hard ceiling on `gold_coverage_avg` (0.675), and the mechanism is monopolisation — on three
baseline rows the top-10 was **ten pages of a single PDF**, so the second gold document was
shut out of the slate entirely rather than ranked just below it. No `RERANK_K` value can
recover a page retrieval never offered.

**The method is the reusable part: the fix was scored before it was written.** Because
`candidate_pages` is stored per row and `gold_rank`/coverage are pure functions of
(pages, gold), the `RETRIEVE_K=50` probe report is a **simulator for any slate policy**.
Re-scoring it offline cost nothing and settled four things before a line of code existed:
cap=4 is the optimum at k=10 (3 and 5 both score lower), a 2× fanout pool is sufficient
(coverage saturates at pool=20 — 25/30/40/50 add nothing), round-robin interleave is far
worse (0.575), and a demote-instead-of-drop variant is *exactly* equivalent (the demoted
copy never re-enters a slate this deep). Every live arm then reproduced its simulated
numbers to four decimal places. Simulate slate policy against a stored deep probe before
running anything.

**Shipped:**
- **`vector_store._diversify(hits, cap, k)`** — pure, dicts in and a subsequence out, so it
  unit-tests with no Qdrant. Applied **after** `search`'s payload/file validation loop, so a
  hit dropped for a missing page image is backfilled from the wider pool instead of
  shrinking the slate — a small robustness win the uncapped path never had.
- **`MAX_PAGES_PER_DOC` (5) and `CANDIDATE_FANOUT` (2.0)**, env-overridable like the other
  eval knobs. `MAX_PAGES_PER_DOC=0` restores the exact pre-diversity behaviour *including*
  the narrower fetch, which is what makes the control arm a genuine control.
- **`RETRIEVE_K` 10 → 12.**
- **Both knobs in `run_eval`'s config snapshot** — without them `diff_reports` printed
  "config changes: none" on precisely the comparison this pass exists to make.

Rejected: Qdrant `query_points_groups(group_by="pdf")`. Server-only (`QdrantLocal` has no
group support), so it would fork the two Qdrant modes inside the one function that already
carries most of that complexity, and it reorders by group-best rather than by score.

**The five arms** (`--retrieval-only`: no API key, no Gemini spend, and deterministic —
re-running an arm reproduced it digit for digit):

| arm | recall@1 | recall@3 | gold in slate | `cand_cov` | pages → rerank |
| --- | --- | --- | --- | --- | --- |
| control, k=10 | 0.7397 | 0.9041 | 0.9863 | 0.700 | 10 |
| cap=4, k=10 | 0.7397 | 0.9041 | 0.9726 | 0.825 | 10 |
| k=12, no cap | 0.7397 | 0.9041 | 0.9863 | 0.800 | 12 |
| **cap=5, k=12** | 0.7397 | 0.9041 | **0.9863** | **0.825** | 12 |
| cap=4, k=12 | 0.7397 | 0.9041 | 0.9726 | 0.850 | 12 |

**Finding 1: widening did most of what the cap was credited with.** `RETRIEVE_K` 10 → 12
alone is worth **0.100 of the 0.125**, at zero recall cost — and it was never tried, because
the attribution pass framed the problem as diversity and went looking for a diversity fix.
The `k=50` probe had the evidence for it all along. When a probe rules a *mechanism* in, it
does not thereby rule out the boring lever that addresses the same symptom; measure both.

**Finding 2: every cap=4 arm costs the same gold page, and it is a single-document row.**
`colpali-avg-ndcg` ("What average nDCG@5 does ColPali achieve across ViDoRe?") is answered
by `colpali.pdf` p7, which is the **5th** colpali page in the ranking — so any cap below 5
evicts it and backfills the slate with six documents that have nothing to do with the
question. This is the cap's cost concentrated in one place: on a single-document question
diversity is pure loss, and 63 of the 83 questions are single-document. `cap=5` is the only
setting that takes the full coverage win with nothing evicted, which is the same
zero-regressions bar the `RERANK_K` decision was adopted on.

**Finding 3: the one coverage regression is real, not an instrument artifact.**
`xdoc-donut-encoder-layoutlm-embeddings` goes 1.0 → 0.5 because `donut.pdf`'s first four
slots (p9, p10, p13, p11) are **all non-gold**: the cap spends the document's quota on wrong
pages and evicts the right ones (p4, p8), while gaining a second `layoutlm.pdf` gold page.
A rank-based cap cannot know which of a document's pages are the useful ones. Six rows gain
coverage against this one loss, so it is a good trade — but it is the honest failure mode of
capping by rank, and it is what a relevance-aware cap would have to beat.

**The judged run: `gold_coverage_avg` moved the full distance.** `degradation` all zeros
(the first attempt died on depleted Gemini credits and was aborted before it wrote anything —
a report that cannot be pinned is not worth 25 minutes).

| metric | swept (k=10, no cap) | **diverse (k=12, cap 5)** |
| --- | --- | --- |
| recall@1 / recall@3 | 0.7397 / 0.9041 | 0.7397 / 0.9041 |
| recall@k / rerank_recall | 0.9863 / 0.9863 | 0.9863 / 0.9863 |
| citation_accuracy | 0.9452 | 0.9315 |
| substring_accuracy | 0.9444 | 0.9444 |
| judge_accuracy / score | 0.9178 / 4.77 | 0.9178 / 4.78 |
| **gold_coverage_avg** | 0.675 | **0.825** |
| candidate_coverage_avg | 0.700 | 0.825 |
| avg_latency_ms | 14996 | 18049 |

**`gold_coverage_avg` 0.675 → 0.825, and it now *equals* `candidate_coverage_avg`.** The
reranker is losing nothing at all: coverage sits at 100% of what retrieval offers, against
96% before. Every point of ceiling the pass bought was converted. Per-row: 7 improved, 1
regressed — and the improvement includes `xdoc-donut-ocr-free-docvqa-questions`, the single
*rerank*-only miss the attribution pass identified, which a deeper slate fixed for free.

**Cost: latency +20%** (15.0 s → 18.0 s), the price of triaging 12 thumbnails instead of 10.
The deterministic arms could not price this; it is the one number that needed the judged run.

**The citation audit** (−0.0137 = one question; 2 regressions, 1 gain, all three cross-doc,
all three checked against `cited` before being counted):
- `xdoc-donut-encoder-layoutlm-embeddings` — **genuine, and the predicted cost.** The cap
  evicted donut's gold pages, so the model saw only layoutlm and *declined* the donut half
  ("The provided pages do not contain information about the visual encoder architecture used
  by Donut") rather than answering from parametric memory. The right failure to have.
- `xdoc-ndcg-cutoffs` — **genuine, and the more interesting one.** Coverage *improved*
  0.5 → 1.0 and the answer is right (judge 5/5, substring True), but it cited `colpali.pdf`
  p2, which `find_in_pdfs.py` confirms mentions ViDoRe and never nDCG in any form. Not an
  under-labeled page. A better slate can still produce a worse citation: the reranker
  ordered a topical-but-unstating page first and the answer step pointed at it.
- `xdoc-siglip-loss-paligemma-resolution` — citation *gained* (`siglip.pdf` p1, gold, against
  `colpali.pdf` p22 before) while the judge fell, because the answer dropped the PaliGemma
  half. Citation quality and answer completeness moved in opposite directions on one row.

**Re-derived gates** (the convention: ~3 questions of slack; one question is 0.0137 on the
answerable denominators, 0.05 on either coverage, 0.1 on abstention):

```bash
PYTHONPATH=. uv run python eval/run_eval.py --judge --gate recall@1:0.68 --gate recall@3:0.86 \
  --gate recall@12:0.94 --gate rerank_recall:0.94 --gate citation_accuracy:0.89 \
  --gate substring_accuracy:0.90 --gate abstention_accuracy:0.90 --gate gold_coverage_avg:0.67 \
  --gate candidate_coverage_avg:0.77 --gate judge_accuracy:0.87
```

Three changes beyond the raised coverage floors. **`recall@10` is now `recall@12`** — the
harness derives `ks = {1, 3, RETRIEVE_K}`, so the old gate name no longer exists in a report
and would fail as a missing metric rather than a regression. `citation_accuracy` drops
0.90 → 0.89 to keep its three questions of slack at the new 0.9315. And
`candidate_coverage_avg` goes 0.65 → **0.77**, still deliberately the tightest gate at ~1
question, because it is pure retrieval and carries no LLM variance.

**Baseline:** `eval/reports/baseline_diverse.json`. Diff the pass against
`baseline_swept.json`, which is the same 83 questions at `RETRIEVE_K=10` with no cap.

**The remaining headroom after this pass.** `candidate_coverage_avg` 0.825 leaves 3.5 of 20
cross-document rows uncovered. `xdoc-splade-vocab-dpr-dense`'s `dpr.pdf` page is outside the
top-50 entirely and needs **query decomposition** (embed each half of a two-part question,
union the candidates) — no slate policy reaches it.
`xdoc-donut-encoder-layoutlm-embeddings` needs a **relevance-aware** cap rather than a
rank-based one, per Finding 3.

---

## Query-decomposition pass (follow-on) ✅ DONE

**The problem.** The slate-diversity pass ended with `gold_coverage_avg` *equal* to
`candidate_coverage_avg` (both 0.825): rerank was losing nothing, so every remaining
point of headroom was retrieval-side. One row was named as unreachable by any slate
policy — `xdoc-splade-vocab-dpr-dense`, whose gold `dpr.pdf` p3 is outside the whole
question's top-**50**. The prescription written down then was query decomposition. This
pass ran it, and the prescription was right.

**Shipped:**
- **`src/query_decompose.py`** — `decompose()` (pure, conservative splitter) and
  `fuse_rrf()` (weighted reciprocal rank fusion). Plain strings and dicts in and out, so
  the whole module unit-tests with no Qdrant, following the `_diversify` precedent.
- **`src/retrieval.py`** — the question → candidates seam, shared by `graph.retrieve_node`
  and `eval.run_eval.run_retrieval_only`. See "the arm that measured itself" below.
- **`vector_store._fetch` / `search_multi`** — one Qdrant round-trip per sub-query, fused
  then `_diversify`d. `search` delegates with a one-element list and is byte-identical:
  fusion over a single ranking preserves both its order and its raw scores.
- **`QUERY_DECOMPOSE` (on), `MAX_SUBQUERIES` (2), `DECOMPOSE_ORIGINAL_WEIGHT` (0)**,
  env-overridable like the other eval knobs and all three in the report's config snapshot.
- **`eval/dataset_paraphrase.jsonl`** — a 12-row hold-out, run via the existing `--dataset`
  flag so the pinned baseline's denominators never move.

### Three findings, in ascending order of how much they cost to learn

**1. The arm that measured itself.** `run_retrieval_only` embedded and searched directly
rather than going through the graph. That duplication was harmless while retrieval was one
embed + one search, and stopped being harmless the moment decomposition made "how a
question becomes candidates" a *policy*: the knob moved the pipeline while the eval went
on measuring the old path. The first treatment arm came back byte-identical to its control
on every metric — which read as "decomposition does nothing" and was in fact "the eval
never ran it." **A config knob that changes a stage is only measurable if the harness and
the pipeline reach that stage through the same seam.** Fixing it also caught two
`test_run_eval` tests stubbing a path that no longer bound, which had been quietly hitting
live Qdrant; the suite went 23s → 5s.

**2. Fusing the whole question beside its halves actively suppresses the fix.** RRF rewards
*agreement*: a page found by two rankings outranks a page found well by one. So including
the whole question gives the query that **cannot** find the second document an equal vote,
and the pages it already liked get double-counted. Measured, `dpr.pdf` p3 — rank **4** in
its own half's results — lost its slate slot to `dpr.pdf` p1 and p9, which the whole
question ranked 11th and 12th. The equal-weight arm bought **zero** coverage for 9 rows of
worsened gold rank. Dropping the whole question from the fusion is what made the pass work.
The design note that predicted keeping it was "strictly additive, therefore safer" was
wrong, and wrong in a way only the arm could show.

A contributing cause worth recording: **`_RRF_K = 60` is calibrated for TREC lists
thousands deep.** Across a 12-page slate it flattens ranks 1–4 to within ~5% of each other,
so mere co-occurrence dominates rank position. A smaller damping constant is an untried
lever if fusion is revisited.

**3. "Outside the top-50" was about a *page*, not a *document*.** `dpr.pdf` has 8 pages in
the whole question's top-50 (ranks 14–41); only the gold page p3 is absent. That distinction
is the whole reason a deeper slate and a looser cap both failed on this row, and `CLAUDE.md`
had it as "document", which implies the opposite fix. Corrected.

### The arms (`--retrieval-only`: no API key, no Gemini spend, deterministic)

| arm | recall@1 | recall@3 | recall@12 | `cand_cov` |
| --- | --- | --- | --- | --- |
| control (off) | 0.7397 | 0.9041 | 0.9863 | 0.825 |
| whole + halves (weight 1) | 0.6986 | 0.8630 | 0.9863 | 0.825 |
| **halves only (weight 0)** | 0.6712 | 0.8493 | **1.0000** | **0.850** |

Unlike the slate pass, these **cannot** be simulated from `probe_k50_retrieval.json`: that
probe is a simulator for slate *policy*, and this changes the *query*, so there is nothing
stored to re-score. Live arms are still free.

### The hold-out — built before the arms, and it earned its keep

Every cross-document row in `dataset.jsonl` is phrased `"<A>, and <B>?"`, so a splitter
keyed on `", and "` would score beautifully and prove only that it had memorised the
dataset. `dataset_paraphrase.jsonl` re-phrases 12 rows across five forms (sentence,
versus, both, compare, semicolon) reusing the same gold.

**The splitter fires on 5 of the 12.** It handles sentence boundaries and "versus"; it does
not handle "For both X and Y…", "Compare X with Y", or a semicolon. That number was
measured *before* the arms and the splitter was deliberately **not** edited afterwards —
tuning it to the hold-out would have converted the hold-out into training data.

On those 5 rows coverage still moved **0.7917 → 0.8333**, same direction and magnitude as
the main slice, and the paraphrased target row `para-splade-vocab-dpr-dense` reproduced the
fix (0.5 → 1.0) on a phrasing the splitter had never been shown. **The win generalises; its
reach is ~40% of naturally-varied phrasings.** That reach, not the coverage number, is what
an LLM splitter would have to beat to justify its extra call.

### The judged run (`degradation` all zeros)

| metric | `baseline_diverse` | **`baseline_decomposed`** |
| --- | --- | --- |
| recall@1 / recall@3 | 0.7397 / 0.9041 | 0.6712 / 0.8493 |
| recall@12 / rerank_recall | 0.9863 / 0.9863 | **1.0000 / 1.0000** |
| citation_accuracy | 0.9315 | **0.9589** |
| substring_accuracy | 0.9444 | **0.9583** |
| judge_accuracy / score | 0.9178 / 4.78 | **0.9452 / 4.85** |
| gold_coverage_avg | 0.8250 | 0.8250 |
| candidate_coverage_avg | 0.8250 | **0.8500** |
| avg_latency_ms | 18049 | 18984 |

**The retrieval-only arms priced a cost the pipeline does not pay.** recall@1 and recall@3
both fell — the halves order the top of the slate worse than the whole question did — and
*every* answer-level metric improved anyway. `RERANK_K=3` picks from a 12-page slate, so
what gates the answer is whether gold is **in** the slate, and `rerank_recall` is now 1.0:
every answerable question's gold page survives into the answer step. Retrieval-precision
proxies are not answer quality, and on this pipeline they can move in opposite directions.

**Audit of the flips** (23 of the 83 questions split, so only those can be affected):
- **2 of the 6 judge flips are on questions that do not split** — `sales-q4-revenue`
  (F→T) and `sales-q2-revenue` (T→F). Decomposition cannot have caused them; they are
  run-to-run judge noise, and they cancel. Useful calibration: **~2 questions of judge
  noise on identical retrieval**, which is why no LLM-dependent gate sits closer than 3.
- **Citation: 4 genuine gains against 2 genuine losses.**
- **`xdoc-splade-vocab-dpr-dense`, the target row, worked** — coverage 0.5 → 1.0, and it
  now cites `dpr.pdf` p3, the gold page, where the baseline cited `colbertv2.pdf` p2 which
  is not in gold at all. But its answer dropped the word "vocabulary" for "sparse
  representations of terms", so `substring_match` and the judge both flipped False. Better
  grounding, weaker wording, on the same row.
- **`xdoc-donut-ocr-free-docvqa-questions` is a real citation regression, checked.**
  `find_in_pdfs.py` confirms `donut.pdf` p8 contains **no** OCR mention, so this is not
  under-labeled gold — it is the failure the slate pass also recorded, where rerank orders
  a topical-but-unstating page first and the answer step points at it.

### Two costs to carry forward

**`gold_coverage_avg` did not follow its ceiling.** It stayed at 0.825 while
`candidate_coverage_avg` rose to 0.850, so the slate pass's headline property — *rerank
loses nothing* — no longer holds: rerank now drops a gold page it was offered. The
attribution pair is doing exactly its job in flagging that, and it is the most concrete
open lead in this file.

**Latency +5.2%** (18.0 s → 19.0 s), from one extra embed and one extra Qdrant query on
the 28% of questions that split. No extra Gemini tokens: the slate stays `RETRIEVE_K` wide
however many sub-queries fed it.

### Re-derived gates

Adopting this **lowered two floors** — that is the real price. `recall@1` and `recall@3`
are genuinely worse, and the guard on retrieval precision is correspondingly weaker; the
trade is that six answer-level metrics are better and `rerank_recall` is perfect.

```bash
PYTHONPATH=. uv run python eval/run_eval.py --judge --gate recall@1:0.63 --gate recall@3:0.81 \
  --gate recall@12:0.95 --gate rerank_recall:0.95 --gate citation_accuracy:0.91 \
  --gate substring_accuracy:0.91 --gate abstention_accuracy:0.90 --gate gold_coverage_avg:0.67 \
  --gate candidate_coverage_avg:0.80 --gate judge_accuracy:0.90
```

**Baseline:** `eval/reports/baseline_decomposed.json`. Diff against `baseline_diverse.json`,
which is the same 83 questions with decomposition off.

### What is left

- **`gold_coverage_avg` vs its ceiling** (0.825 against 0.850) — rerank is losing a row it
  is offered, for the first time since the slate pass closed that gap.
- **The splitter reaches ~40% of phrasings.** "For both X and Y…", "Compare X with Y" and
  semicolon-joined questions are not split. This is the measured bar an LLM splitter must
  clear, and the reason to consider one is *reach*, not accuracy on what it already splits.
- **`_RRF_K = 60` is untuned** for slates this shallow (see finding 2).
- **`xdoc-donut-encoder-layoutlm-embeddings`** still needs a **relevance-aware** cap — a
  rank-based one spends that document's quota on four non-gold pages. Unchanged by this pass.

---

## Batch-embedder pass (follow-on) ✅ DONE

**The lever.** Ingest is the slowest thing this project does. Nothing had ever measured
*which stage* was slow, so `scripts/profile_ingest.py` was written to time all four
(render / save / embed / upsert) before optimising any of them.

The answer is unambiguous, and it is the only number here that generalises:

| stage | per page | share |
|---|---|---|
| render (poppler) | 0.123 s | 1.5% |
| save PNG | 0.047 s | 0.6% |
| **embed (ColQwen2 forward)** | **8.242 s** | **98.0%** |
| upsert | not measured (`--store` opt-in) | — |

So the forward pass is the only stage worth touching, and batching it is the only thing
that helps it. `embed_image` became `embedder.iter_embedded`, a generator yielding
`(start_index, vectors)` per batch, and `ingest_pdf` consumes it.

### The finding that changed the pass

`--verify-equivalence` — the check that licenses keeping `EMBED_BATCH_SIZE` out of
`EMBED_VERSION` — **failed on the first run**, at `max_abs_delta 0.451` against a 0.01
tolerance, with every patch count matching. Padding was not the cause. Six experiments
pinned it down:

| experiment | result | rules out |
|---|---|---|
| same call twice, batch 1 | delta **0.000000** | model nondeterminism |
| batch 1 vs 2 / 4 / 8 | 0.430 / 0.411 / 0.411 | — |
| per-page breakdown | **page 1 only**; pages 2-4 bit-identical | uniform float noise |
| reorder to `[p2,p1,p3,p4]` | **slot 0** corrupt, whichever page sits there | page content |
| **same page twice in one batch** | slot0 vs slot1 = **0.378** | everything else |
| **CPU float32**, batch 1 vs 2 | **0.00000000** | the batching code itself |

The same image, in the same batch, produces different vectors at slot 0 than at slot 1.
That is a computation bug in the backend, not precision loss — and it is **MPS + bfloat16
specific**. `attn_implementation` is not the lever (`eager`, `sdpa` and the default all
corrupt); **dtype is** (MPS float32 and CPU float32 are both exact). Compare
pytorch/pytorch#162592 and #163597, on torch 2.11.0 / transformers 5.3.0.

**What that would have shipped.** One page in every `EMBED_BATCH_SIZE` — 25% of the corpus
at the default of 4 — silently poisoned, with no error, no NaN, correct patch counts, and a
`content_hash` recording the document as current so no later sync would repair it. It would
have degraded retrieval quietly and permanently. **No test that stubs the model could have
caught it**, which is the argument for the gate existing at all.

### The fix

**Batching is disabled on MPS** — `embedder._batching_is_supported`, a device check
evaluated once per process. Deliberately the blunt version rather than a runtime
measurement, and the two costs of that are known and accepted:

- **MPS float32 would batch correctly and is refused anyway.** Only bf16 is affected, and
  bf16 is what `_device_and_dtype` selects on MPS, so in this codebase the device check and
  the dtype check pick out the same configuration.
- **A fixed torch will not re-enable batching by itself.** Someone has to delete the check.
  `--verify-equivalence` is how you would discover it is safe again; it is the thing to
  re-run after any dtype, device, model or torch change.

A runtime self-check (embed a batch's first page alone, compare, fall back on disagreement)
was built and then removed as not worth its complexity. One measurement from it is worth
keeping, because it would trap anyone who rebuilds it: **it cannot be probed with a cheap
synthetic image.** At 28-112 px the corruption does not reproduce at all — sequence lengths
15-27 come back clean, against ~755 for a rendered page — so a throwaway-image probe returns
a confident all-clear on exactly the configuration it exists to catch. It has to use real
pages, which is most of why it cost more than it was worth.

**Also rejected: a sacrificial slot-0 image.** Prepending a throwaway page to every batch
would put real pages at slots >=1, which measured bit-identical to solo, and would keep
batching on MPS for ~20-25% overhead. Rejected because it stakes index correctness on an
unfixed upstream bug corrupting *exactly* slot 0 and nothing else — an assumption with no
upstream guarantee, whose violation is again silent.

### The other defect, found reading the interrupted code back

`ingest_pdf` pre-chunked pages into `EMBED_BATCH_SIZE` windows and called the embedder once
per window. The OOM backoff halves and *keeps* the smaller size — but only for the life of
one call, so per-window calls threw that away and re-paid the failed forward pass on **every
window**, precisely the failure the docstring claimed to avoid. Fixed by making the seam a
generator spanning the whole document. `tests/test_ingest.py::test_the_whole_document_goes_to_one_embed_generator`
is the guard, and it fails (`[4, 4, 2] != [10]`) against the old shape.

### What this pass actually delivers

**Correctness on Apple Silicon; throughput only on CUDA.** With batching disabled there,
every arm of the sweep ran at an effective batch of 1:

| requested | actual | embed/page | pages/min |
|---|---|---|---|
| 1 | 1 | 8.242 s | 7.13 |
| 2 | **1** | 9.838 s | 6.00 |
| 4 | **1** | 12.074 s | 4.90 |
| 8 | **1** | 11.856 s | 4.99 |

Read that table as a **noise floor, not a regression**: all four arms did identical batch-1
work and still spread 8.242 → 12.074 s/page (+46%) through thermal drift over a ~20-minute
run. Nothing under a ~1.5x claim is measurable on this box. The profiler now records
`effective_batch_size` and `requested_size_honoured` per arm precisely so a degenerate arm
cannot be read as a speedup number.

`EMBED_BATCH_SIZE` therefore ships at **4, untuned and documented as such** — it engages only
where batching verifies, which today means CUDA, and this machine cannot measure it.

**Baseline:** `bench/reports/ingest_baseline.json` (pinned; MPS + bf16, so its arms are
degenerate by construction — it is evidence for the 98% share and for the guard firing on a
real backend, not for what batching buys). A CUDA run would supersede it.

### What is left

- **`EMBED_BATCH_SIZE` has never been measured on a backend that honours it.** The sweep on a
  CUDA box is the missing experiment; the report already carries the fields to make it honest.
- **MPS float32 is correct and could batch** — untested, and unlikely to pay on a 16 GB box:
  ~8 GB of fp32 weights before activations, and fp32 MPS ops run ~2x slower than bf16, so
  batching would have to win >2x just to break even.
- **The equivalence gate is opt-in.** It needs the real model, so it cannot join `pytest`.
  It is the one thing to run after any change to dtype, device, model or torch version. Its
  verdict is three-valued: `corrupt` (exit 1, the only failure), `equivalent`, and
  `not_applicable` (exit 0 - the backend never batched, so the stored vectors are right but
  batching is untested). **On MPS it always returns `not_applicable`**, because batching is
  disabled there by design. `tests/test_profile_ingest.py` pins that distinction: a gate that
  ANDs "did batching run" into "did batching change the vectors" reports the correct
  configuration as corruption on every Apple Silicon machine.
- **`EMBED_VERSION` does not capture processor version, device, or backend-specific behavior**
  (found while verifying this pass, and *not* caused by it). Re-embedding `sales_report.pdf`
  and comparing against the vectors Qdrant already holds gives a max delta of **0.00125** —
  and `git show HEAD:src/embedder.py`'s `embed_image` reproduces that same delta exactly, while
  the new path is bit-identical to it (`0.00000000`). The drift is between the environment now
  and the one that built the index: transformers 5.3.0 warns that `Qwen2VLImageProcessor` "is
  now loaded as a fast processor by default … may produce slightly different outputs". So the
  fingerprint's blind spot is wider than the model-or-DPI case it was designed for: a
  transformers upgrade silently changes embeddings while `content_hash` and `embed_version`
  both still match, and no sync repairs it. Similarly, device/dtype changes (e.g., switching
  from MPS to CUDA, or upgrading torch/transformers) are not tracked. Harmless at this
  magnitude (0.001 on unit-norm vectors; the retrieval eval is unchanged to four decimals),
  but a larger processor or backend change would not announce itself either.

  **Remediation:** After any change to transformers version, torch version, device, dtype, or
  `COLPALI_MODEL` beyond what `EMBED_VERSION` already captures, run `profile_ingest.py
  --verify-equivalence` to check for drift. If vectors differ beyond the tolerance, update
  `EMBED_VERSION` (e.g., append a numeric suffix) to force a full reindex, then run
  `PYTHONPATH=. uv run python -m src.ingest` to rebuild the corpus. Document the required
  reindex in deployment notes when changing these settings.

---

## Ingest-throughput pass (follow-on) ✅ DONE

Work on branch **`perf/ingest-throughput`**. The batch-embedder pass left ingest
effectively unimproved on Apple Silicon — batching is disabled on MPS by design, so every
page still went through the model alone at ~8 s, and the 363-page corpus took ~38 minutes.

**The finding that reframed the pass.** That pass measured *which stage* was slow (embed,
98%) but never measured *inside* it. Splitting embed into its three sub-stages changes the
conclusion: page-at-a-time is not what makes ingest slow, **755 visual tokens per page** is.

| stage | per page | on the GPU's thread? |
|---|---|---|
| render (poppler) | 0.132 s | yes |
| **preprocess (CPU)** | **0.304 s** | yes — was invisible inside "embed" |
| **forward (GPU)** | **5.451 s** | yes |
| decode (GPU→CPU) | 0.046 s | yes |
| save PNG | 0.044 s | yes |
| **upsert** | **0.264 s** | yes — and had *never* been measured |

The upsert is the sharper indictment: `profile_ingest.py`'s `--store` was opt-in, so the
pinned baseline carried `upsert_measured: false` through an entire ingest-optimisation pass
while `upsert_pages` sat inline in the embed loop. It is measured by default now.

### The lever: `EMBED_VISUAL_TOKENS`

The checkpoint caps pages at 768 visual tokens (`max_pixels=602112`), and
`ColQwen2Processor.from_pretrained` accepts a `max_num_visual_tokens` kwarg the code never
passed. ViT attention is quadratic in patch count, so the budget pays **superlinearly**:

| budget | patches | forward | pages/min | speedup |
|---|---|---|---|---|
| **768 (checkpoint default)** | 755 | 6.48 s | 9.3 | 1.00× |
| 512 | 486 | 3.18 s | 18.9 | **2.04×** |
| 384 | 385 | 2.60 s | 23.1 | 2.49× |
| 256 | 263 | 1.64 s | 36.6 | 3.95× |

**Two levers that look obvious and do nothing**, recorded so they are not retried:

- **`RENDER_DPI` cannot speed up embedding.** A 150-DPI page is 1275×1650 = 2.1 MP and
  `smart_resize` hands the model 672×868 — it already reads pages at ~79 DPI equivalent. DPI
  buys render time and disk, and the page PNGs still need it for the full-res answer step.
- **dtype is not it.** fp16 vs bf16 measured 6.5 s vs 6.5 s, inside this box's ~46% noise floor.

**MRL was investigated and does not apply, twice over.** ColQwen2-v1.0 is a LoRA adapter over
`vidore/colqwen2-base` with a fixed-width `custom_text_proj` and no matryoshka config, so
truncating dimensions would degrade unpredictably rather than gracefully. More fundamentally
it targets the 1536→128 projection, not the 2.25B backbone, so it cannot move the forward
pass at all — and as a storage lever `BinaryQuantization` already beats it 32× to 2–4×.

### The arm: 512 tokens ✗ REJECTED (and *why* is the useful part)

`COLLECTION_NAME` became env-overridable so the arm could be built and scored without
destroying the pinned index. The control re-scored `pdf_pages` under the new code and
reproduced `baseline_decomposed.json` exactly — which is also the proof that the pipelining
below changed no vectors.

| metric | 768 (control) | 512 | Δ |
|---|---|---|---|
| recall@1 | 0.6712 | 0.6438 | −0.0274 |
| recall@3 | 0.8493 | 0.8082 | −0.0411 |
| **recall@12** | **1.0000** | **0.9315** | **−0.0685** |
| candidate_coverage_avg | 0.8500 | 0.8500 | 0.0000 |

Rejected under the adoption rule (quality first, zero regression), and it fails the pinned
`recall@12:0.95` gate outright. **But the degradation is not uniform, and its shape is the
finding:** 10 rows improved and 5 lost gold from the slate entirely — and **4 of those 5 are
`table` rows** (`attn-big-params`, `colpali-avg-ndcg`, `colpali-ndcg-ai-task`,
`colpali-ndcg-tabfquad`). At 486 patches the model can no longer resolve digits in a dense
numeric table. **Dense-table reading is precisely what pays for the speed**, so a corpus of
prose would likely take this trade happily — which is exactly why the knob ships rather than
the number.

384 and 256 were **not run**: they strictly reduce the information reaching the model, so
they cannot pass a zero-regression rule that 512 already fails, and they would hit those same
table rows harder. `eval/reports/vt_control.json` and `vt512.json` are kept as the evidence.

**Shipped opt-in.** `EMBED_VISUAL_TOKENS` unset means "the checkpoint's own budget", which is
what every stored vector was built with — so adding the knob did not invalidate the index.
Setting it, even to 768, changes `EMBED_VERSION` and re-embeds: erring toward a needless
re-ingest over silently keeping vectors built at a different budget. It goes *into*
`EMBED_VERSION` for the exact reason `EMBED_BATCH_SIZE` stays out.

### Getting the CPU off the GPU's thread — worth less than its serial cost

`iter_embedded` now preprocesses the next batch while the GPU runs the current one, and
`ingest._StoreWorker` takes the PNG save and the Qdrant upsert onto a worker thread. Both are
vector-identical (`--verify-equivalence` reports `max_abs_delta 0.0`).

Measured as **GPU-busy fraction, not s/page** — s/page spreads 50% *within* a single arm
here, while the ratio survives thermal drift because both arms run the identical forward
pass. Three interleaved rounds:

| arm | GPU idle | rounds won vs serial |
|---|---|---|
| serial | 9.7% | — |
| store worker only | 10.0% | **0 of 3** |
| + preprocess lookahead | **7.0%** | **3 of 3** |

0.61 s/page of serial work bought only 2.7 points of GPU idle, because **background Python
threads contend with the GPU dispatch thread for the GIL**. `build_point` is pure-Python, so
the store worker's GIL traffic costs the main thread about what moving the work off it saves
— visible as the main thread's own preprocess inflating 0.177 s → 0.313 s when only that
changed. Kept anyway, because its share grows as the forward pass shrinks (0.31 s against
5.45 s is 6%; against 2.7 s it is 11%), and flagged in its docstring as the piece to
re-measure and delete if it stays flat.

**A deadlock found by reading the code back, not by testing it.** The store worker drains its
queue to the sentinel on failure, so a producer blocked on a full bounded queue is released
rather than waiting on a dead worker. But the sentinel arrives exactly once, and the
*remainder* flush — the tail pages past the last full `UPSERT_BATCH_SIZE` — runs after it has
been consumed. A failure there sent the worker into a drain that could never terminate and
hung the whole ingest, on precisely the path the drain was added to protect, for every
document whose page count is not a multiple of 8. The test runs the ingest on a thread with a
join timeout so a regression fails instead of wedging the suite.

### Batching on MPS: the corruption is fixable, the batching still is not ✗ REJECTED

The bf16 bug corrupts **only slot 0**, so batching can be recovered by prepending a
throwaway page and discarding output index 0. Verified directly, using a duplicate of the
batch's first real page so the padding cannot shift:

```
naive batch of 3 vs solo:   slot 0 delta 0.411133  <-- CORRUPT
                            slot 1 delta 0.000000
                            slot 2 delta 0.000000
sacrificial pad, slot 0 discarded:  all three pages delta 0.000000
```

**So the correctness objection is answerable — and it does not matter, because batching is
slower than batch-1 on this box anyway**, before paying anything for the wasted slot. Median
of three interleaved rounds, per *useful* page:

| batch | median s/page | vs batch 1 | wasted slot |
|---|---|---|---|
| **1** | **8.367** | **1.000×** | — |
| 2 | 11.539 | 0.725× | 33% |
| 4 | 10.921 | 0.766× | 20% |

Batch 1 won 2 of 3 rounds outright. It lost round 3, where it measured 17.190 s/page — but
that round inflated *all three* arms, so it is thermal drift rather than a real crossover;
this is the same 46% within-arm spread that forced the GPU-idle metric above.

Note what this does *not* establish: whether a 755-token forward already saturates the
device, or whether larger batches hit memory pressure on 16 GB, is not separated by this
measurement — only that there is no throughput to win here. Nothing was implemented.
`_batching_is_supported`'s blunt MPS device check turns out to cost this hardware nothing,
which is a stronger justification than the one it shipped with.

### What is left

- **`bench/reports/ingest_baseline.json` was NOT re-pinned, and still predates the sub-stage
  split.** Every attempt to re-pin landed on a box running OneDrive Sync Service at 47–97%
  CPU, which inflated preprocess from 0.304 s to 0.965 s on the same eight pages. Pinning
  that would make every future run look faster than it is. `ingest_contended.json` is the
  new-format run, stamped `measurement_quality: CONTENDED`, kept only to exercise the added
  fields. **Re-pin from a quiet box** — this is the one piece of the pass left undone, and
  the sub-stage figures quoted throughout this section come from the cleaner earlier runs.
- **`EMBED_VISUAL_TOKENS` has never been swept on a corpus that is mostly prose.** The 512
  arm lost 4 table rows and gained 10 others; a corpus without dense numeric tables is the
  case where this knob is free speed, and this eval cannot see it.
- **A CUDA box would change three conclusions at once** — batching, the GIL contention, and
  the token-budget trade all measured against a saturated MPS device.
- **Escaping the GIL for preprocess** with a process pool rather than a thread pool, at the
  cost of pickling ~5 MB of `pixel_values` per batch. Worth it only if a profile still shows
  preprocess inflating the measured forward time.
- **The store worker is on probation** — 0 of 3 rounds today, kept on the argument that its
  share grows as the forward pass shrinks. Re-measure it; delete it if it stays flat.
