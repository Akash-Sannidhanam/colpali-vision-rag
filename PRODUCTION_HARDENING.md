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
rerank-only, 0 mixed** — the reranker preserved coverage on every row where retrieval
offered both documents, losing exactly one. (That no row is the mixed case is why the
two-case reading this section originally shipped with reached the right answer despite
being wrong in general; the third case is now covered by a test rather than by luck.) So
`candidate_coverage_avg` is a hard ceiling on `gold_coverage_avg`, and at 0.675 vs 0.700
coverage already sits at **96% of what retrieval offers**. This retroactively explains why
the `RERANK_K` decision's pre-registered metric did not fire: no value of `RERANK_K` could
have moved it.

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
