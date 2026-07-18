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
| 2 | Observability (request IDs, node timing, token/cost, tracing) | ⬜ Pending |
| 3 | Warm serving + UI (FastAPI + Streamlit) | ⬜ Pending |
| 4 | Evaluation (retrieval + answer-quality suite) | ⬜ Pending |

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

## Phase 2 — Observability ⬜

- **Structured logs across the pipeline** — a per-query `request_id` (uuid4) bound
  via a `contextvar`; wrap each `graph.py` node to log start/end + `latency_ms`.
  Keep the user-facing CLI `print()`s; logs are the machine-readable layer.
- **Gemini token/cost accounting** — already emitted per call by
  `gemini_client._log_usage` (tagged by `purpose`); surface per-query totals once
  the calls are routed through it (Phase 1).
- **LangSmith tracing (opt-in, env only)** — document `LANGSMITH_TRACING` /
  `LANGSMITH_API_KEY` in `.env.example` + README; LangGraph emits traces natively,
  no code change.

**Files:** `src/graph.py`, `src/main.py` (bind `request_id` in `run_query`),
`.env.example`, `README.md`.
**Verify:** one query → JSON logs carry a shared `request_id`, per-node `latency_ms`,
and per-call token counts for both `rerank` and `answer`.

---

## Phase 3 — Warm serving + UI ⬜ (the biggest lever)

- **`src/server.py`** *(new)* — FastAPI app. **Lifespan warmup** loads ColQwen2 once
  (a tiny dummy `embed_query` populates `embedder._model`/`_processor`) and opens the
  Qdrant client, so the ~2B cold start is paid once at boot, not per query.
  Endpoints: `POST /query {question}` → `run_query()` result + crop/annotated images
  (static route under `page_images/` or base64); `GET /health` → model-loaded +
  Qdrant `ping()`; optional `POST /ingest` (upload a PDF). Single worker (one
  GPU-resident model) with async handlers for concurrency — document this.
- **`ui/app.py`** *(new, the `ENHANCEMENTS.md` "highest-value next step")* — a thin
  Streamlit page: question box → calls `/query` → renders the **answer, the annotated
  page, and the cropped slice inline** in the browser. Replaces the macOS-only
  `_open_file`.
- **Deps** — add `fastapi`, `uvicorn[standard]`, `python-multipart`; `streamlit`
  (+ `requests`/`httpx` for the UI→server call) in a `ui` dependency-group.

**Files:** `src/server.py`, `ui/app.py`, `pyproject.toml`, `README.md`.
**Verify:** `uvicorn src.server:app`; first `/query` is warm (no model reload in
logs); hit it twice → model loads once. Load the Streamlit UI, ask a
`sales_report.pdf` question, see the annotated page + crop inline.

---

## Phase 4 — Evaluation ⬜ (the regression guard)

- **`eval/dataset.jsonl`** *(new)* — a small labeled set over the already-shipped
  43-page corpus (`attention.pdf`, `colpali.pdf`, `sales_report.pdf`): each row a
  question + gold `{pdf, page}` and an optional expected-answer substring.
  Formalizes the "validated on 43 pages" result.
- **`eval/run_eval.py`** *(new)* — runs each question and scores:
  - **Retrieval recall@k** — is the gold page in the Qdrant top-`RETRIEVE_K`? in the
    reranked top-`RERANK_K`? (retrieval-only metrics need no Gemini call, so they run
    cheaply/offline against the embedded `qdrant_data/` store).
  - **Citation correctness** — did `answerer`'s `source_page` map to the gold page?
  - **Answer quality** — substring match, plus optional **LLM-as-judge** (Gemini
    scores the answer against the reference).
  - Emits a scored JSON report + a printed table. Re-run after changing DPI,
    `RERANK_K`, or the model to *prove* no regression.

**Files:** `eval/dataset.jsonl`, `eval/run_eval.py`.
**Verify:** `uv run python eval/run_eval.py` prints recall@k + citation + answer
scores; retrieval-only mode runs with no `GEMINI_API_KEY`.

---

## Out of scope (natural follow-ons, not in this pass)

Security / input validation (PDF size/page caps, Qdrant auth/TLS, query length
limits), scaling/perf (batch the embedder — it embeds one page at a time today —
query-result cache, incremental content-hash ingest), and packaging/CI (app
Dockerfile on a GPU base, GitHub Actions with ruff/mypy). The warm server (Phase 3)
makes the app Dockerfile the obvious next step if this later targets a real
deployment.
