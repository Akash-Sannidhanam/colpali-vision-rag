# ColPali Vision RAG

Retrieval-augmented QA over PDFs that treats each page as an **image** rather than text, so charts, tables, and scanned documents work without any OCR or text layer.

It retrieves pages with [ColQwen2](https://huggingface.co/vidore/colqwen2-v1.0) (a ColPali-family vision retriever) + Qdrant, has Gemini **rerank** the candidates down to the pages that actually matter, then asks Gemini to answer the question. The twist: Gemini also returns a **bounding region** for where it read the answer, which is **cropped out of the page and shown to you**, so every answer comes with the exact slice of the chart or table it came from.

```
                                                            ┌─ answer text: "180"
question ─▶ retrieve ─▶ rerank ─▶ answer ─▶ highlight ──────┼─ crop:      page_1_crop.png      ◀─ the exact slice
           (ColQwen2   (Gemini    (Gemini,   (crop the      └─ annotated: page_1_annotated.png  ◀─ box drawn on page
            + Qdrant,   picks the   structured  cited box)
             top-10)    top 2)      output)
```

## Why

Vision RAG can *read* a chart, but a plain text answer ("180") gives the reader no way to check it. Here the model reports **where** it looked, and the pipeline crops that region from the page PNG it already stored during ingestion, turning the answer into a visual, verifiable citation.

## Requirements

- **Python ≥ 3.13** and [**uv**](https://docs.astral.sh/uv/)
- **Docker** (runs the Qdrant vector database via `docker compose`)
- **Poppler** (for `pdf2image` page rendering)
  - macOS: `brew install poppler`
  - Debian/Ubuntu: `sudo apt-get install poppler-utils`
  - (auto-detected on `PATH`; override with the `POPPLER_PATH` env var)
- A **Gemini API key** ([Google AI Studio](https://aistudio.google.com/apikey))
- First run downloads the ColQwen2 weights (~2B params) from Hugging Face. Uses Apple **MPS** on macOS and **CUDA 12.8** wheels on Linux/Windows automatically.

## Setup

```bash
git clone https://github.com/Akash-Sannidhanam/colpali-vision-rag.git
cd colpali-vision-rag
uv sync

# add your key + point the app at the Qdrant server
cp .env.example .env      # then edit GEMINI_API_KEY

# start Qdrant (dashboard at http://localhost:6333/dashboard)
docker compose up -d
```

`.env` is gitignored, so your key stays local. It holds `GEMINI_API_KEY` and
`QDRANT_URL=http://localhost:6333`. Leave `QDRANT_URL` unset to skip Docker and use
the embedded on-disk store instead — handy for quick local runs.

## Usage

```bash
# 1. Generate the sample PDF (a bar chart + a sales table, pure pixels, no text layer)
uv run python scripts/make_sample_pdf.py

# 2. (optional) Fetch the eval corpus — 16 arXiv papers pinned by sha256, ~320 pages
uv run python scripts/fetch_eval_corpus.py          # needed only to run eval/

# 3. Ingest: render pages → embed with ColQwen2 → store in Qdrant
PYTHONPATH=. uv run python src/ingest.py            # indexes everything in pdfs/
#   or point at specific files:  ... src/ingest.py path/to/doc.pdf
#   ingest is incremental — re-running only embeds documents whose bytes (or the
#   model / render DPI that produced them) changed, so an unchanged corpus is a no-op
PYTHONPATH=. uv run python src/ingest.py --rebuild   # force a full atomic re-index

# 4. Ask a question
PYTHONPATH=. uv run python src/main.py "What was the Q4 revenue in the chart?"
```

The repo ships a small starter corpus in `pdfs/` — the generated sales report plus
two arXiv papers (*Attention Is All You Need* and *ColPali*, ~43 pages total) — so
the rerank step has a real 10-candidate pool to narrow out of the box. Drop your own
PDFs into `pdfs/` and re-run the ingest to index them too.

Step 2 is only needed to run the eval. It adds 16 more papers (~320 pages) chosen to
be *confusable* with the two gold documents, because a 43-page corpus is too small to
measure retrieval on at all: `RETRIEVE_K=10` over 43 pages returns a quarter of the
index every query, so recall@10 cannot be anything but 1.0. See
[Evaluation](#evaluation). Those PDFs are not committed — `eval/corpus_manifest.json`
pins each by sha256 and the fetch script verifies them.

### Example output

```
============================================================
RETRIEVED PAGES
============================================================
sales_report.pdf- page 1 (score 13.8517)
sales_report.pdf- page 2 (score 7.9038)

============================================================
ANSWER
============================================================
180
============================================================

============================================================
SOURCE REGION
============================================================
From sales_report.pdf - page 1
crop:      page_images/crops/sales_report_page_1_crop.png
annotated: page_images/crops/sales_report_page_1_annotated.png
============================================================
```

The `crop` is a tight slice around the answer; the `annotated` page is the full page with the region outlined so you can see where it sits. On macOS the crop opens automatically in Preview.

Try `"Which region had the highest growth?"` to hit the table page instead.

`RETRIEVED PAGES` lists the pages kept *after* reranking. Retrieval pulls `RETRIEVE_K` (10) candidates from Qdrant and the rerank step narrows them to `RERANK_K` (2) before the answer step runs; on a corpus of ≤2 pages there is nothing to trim, so rerank passes straight through. The shipped ~43-page corpus exercises the full 10→2 path — e.g. `"What was the Q4 revenue in the chart?"` still lands on `sales_report.pdf` even though it is now 2 pages among ~43.

## Serving (HTTP API + UI)

Beyond the CLI, the pipeline runs as a warm **FastAPI** service with a **React** UI on
top. The service loads the ~2B ColQwen2 model **once at startup** (not per query), so
after boot every request is warm.

### API server

```bash
# warm the model + Qdrant once, then serve on http://127.0.0.1:8000
PYTHONPATH=. uv run uvicorn src.server:app --host 127.0.0.1 --port 8000
```

Run a **single worker** — the one GPU-resident model is shared and serialized behind a
lock, so `--workers >1` would load N copies and break that assumption. On startup you
see the model load once and a `server warm` log line; the live OpenAPI schema is at
`/docs`.

| Method & path | What it does |
|---|---|
| `POST /query` `{question}` | Answer + citation (with `box`) + the used pages + crop/annotated images + a per-request `meta` (request_id, latency, tokens, cost, and a per-stage breakdown). Add `?inline=true` to also get images as base64 data-URIs (for a sandboxed UI); the default returns `/images/...` URLs. |
| `POST /heatmap` `{question, pdf, page_number}` | Per-patch **MaxSim heatmap** for one page — an `n_x × n_y` grid of query→page match strengths in `[0,1]`. Powers the viewer's **"why this page?"** toggle (which patches the query lit up, vs. the crop's *where the answer was read*). On-demand: it recomputes two forward passes on the model lock, so it's a separate call, not part of `/query`. |
| `GET /health` | `model_loaded` + Qdrant reachability. `503` when Qdrant is unreachable. |
| `GET /corpus` | Indexed documents + page counts (powers the UI's corpus rail). |
| `POST /ingest` (multipart PDF) | Render → embed → index an uploaded PDF. Blocking and long-running — it holds the model lock for the whole build. Only the uploaded document is embedded; re-uploading an unchanged one is recognised and costs nothing. |
| `DELETE /corpus/{pdf}` | Remove a document completely — its vectors, page images, crops, and the stored PDF. `404` when it isn't indexed. Takes no model lock, so it stays responsive during a query or ingest. |
| `GET /images/...` | Static page / crop / annotated PNGs. |

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What was the Q4 revenue in the chart?"}' | jq .answer   # "180"
```

#### Auth and rate limiting

Set `API_KEY` and every endpoint above requires it in an `X-API-Key` header. **Leaving
it unset disables auth entirely**, which is the default so a local run needs no setup —
the server logs a warning at boot when it starts up open.

```bash
API_KEY=$(openssl rand -hex 24) PYTHONPATH=. uv run uvicorn src.server:app
curl -s localhost:8000/corpus                          # 401
curl -s -H "X-API-Key: $API_KEY" localhost:8000/corpus # 200
```

Two endpoints stay open by design:

- **`GET /health`** — so an orchestrator can probe liveness without holding the secret.
- **`GET /images/...`** — because `<img src>` cannot send a custom header. Page and crop
  PNGs are all it exposes, and their paths are only discoverable through an
  authenticated `/query`. **If your pages are themselves sensitive, this is the gap to
  close** (put the deployment behind a proxy that authenticates, or serve images as
  data-URIs with `?inline=true`).

Requests are also rate limited per client IP — a sliding window, counted in-process
(the server is single-worker by design, so one process sees everything). Exceeding it
returns `429` with a `Retry-After` header. The limit is applied **before** the key
check, so unauthenticated requests are throttled too and key guessing isn't free.

| Setting | Default | Notes |
|---|---|---|
| `SERVER_HOST` | `127.0.0.1` | uvicorn bind host (the `python src/server.py` runner) |
| `SERVER_PORT` | `8000` | uvicorn bind port |
| `API_KEY` | *(empty)* | shared secret required in `X-API-Key`. **Empty disables auth** — set it for anything reachable beyond localhost |
| `RATE_LIMIT_PER_MINUTE` | `30` | per-IP cap on the query/read endpoints; `0` disables |
| `RATE_LIMIT_INGEST_PER_HOUR` | `10` | per-IP cap on `/ingest` and `/ingest/stream`, on top of the per-minute one; `0` disables |
| `TRUST_PROXY_HEADERS` | `false` | read the client IP from `X-Forwarded-For`. Only enable behind a trusted proxy — otherwise any client can forge it and get a fresh rate-limit bucket |
| `CORS_ALLOW_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | comma-separated browser origins allowed to call the API; `*` allows any. Irrelevant in the Docker deployment, where the UI is served from the same origin |
| `MAX_UPLOAD_MB` | `50` | reject larger PDF uploads to `POST /ingest` |

### UI (`ui/`)

A React + Vite single-page app — a three-column workspace (corpus rail · conversation ·
document viewer) that renders each answer with its **visual citation**: the cited page
with the bounding box drawn over it, the cropped slice, and the reranked-candidate rail,
plus a "how this was answered" per-stage trace. A **"why this page?"** toggle on the viewer
overlays the MaxSim patch heatmap (via `POST /heatmap`), tinting the patches the query
matched — the retrieval-side complement to the answer crop. In development it runs as its
own dev server against the API (two processes: the API on `:8000`, the UI on `:5173`); in
the Docker image it is compiled to static assets and served by the API itself on one origin.

```bash
cd ui
npm install
npm run dev            # http://localhost:5173  (expects the API on :8000)
```

That's the **dev** setup — two processes, cross-origin, which is why the API allows the
Vite dev origin via CORS. A **production build defaults to same-origin relative URLs**
instead, because the deployed shape is FastAPI serving the built bundle itself (see
Deployment below). `VITE_API_BASE` overrides either (e.g.
`VITE_API_BASE=http://host:8000 npm run dev` to point a dev UI at a remote backend).

`npm run typecheck` and `npm run test` cover the UI's pure logic (the
`citation.box → overlay` math and 1-based page resolution); CI runs both plus the build
on every PR.

### Deployment (Docker)

The `Dockerfile` packages the **API and the UI together**: a `node:22-slim` stage
compiles `ui/` to static assets, and FastAPI serves them at `/` alongside the API. One
container, one origin — so there is no separate web server to run and CORS never enters
the picture. The Python side is a multi-stage `uv` build on a slim base; on Linux it
pulls the CUDA 12.8 torch wheels, so the image is GPU-capable with `--gpus all` and
**auto-falls back to CPU** when no GPU is present. Poppler is included; it runs as a
non-root user and serves on `0.0.0.0:8000`.

The whole stack, which is the recommended path — the `app` service is wired to the
`qdrant` service and passes secrets through from your environment:

```bash
GEMINI_API_KEY=... API_KEY=... docker compose up   # then open http://localhost:8000
docker compose up -d qdrant                        # Qdrant only (run the app on the host)
```

`http://localhost:8000` is the whole product: the UI loads there and calls the API on
its own origin. With `API_KEY` set it prompts for the key on first load and keeps it for
that browser tab (never baked into the bundle — that JS ships to every visitor).

The image alone, if you're wiring it into something else:

```bash
docker build -t vision-rag .
docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e API_KEY=$API_KEY \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -v vision-rag-hf:/home/appuser/.cache/huggingface \   # persist the model download
  vision-rag                                            # add --gpus all on a GPU host
```

First boot downloads the ~2B ColQwen2 model into the mounted HF cache (subsequent boots
are warm); `/health` returns `503` until the model is loaded and Qdrant is reachable.
Qdrant must be reachable at `QDRANT_URL` — the model isn't baked in, and Qdrant runs
separately (see the compose file).

On a GPU host, uncomment the `deploy.resources` block in the `app` service (needs the
NVIDIA container toolkit).

**Before exposing it beyond localhost:** set `API_KEY`, keep the single worker, and
terminate TLS at a proxy in front (the app speaks plain HTTP, so an `X-API-Key` on an
unencrypted hop is readable in transit). Note the `/images` exemption described above.

## How it works

1. **Retrieve** (`src/embedder.py`, `src/vector_store.py`): the query is embedded into ColQwen2's token-level multivectors and matched against per-page multivectors in a Qdrant server collection ranked by **MaxSim**. The vectors are **binary-quantized** (128-d → 128 bits, 32× smaller) and kept in RAM for a fast first pass; the top hits are then **rescored** against the full-precision vectors on disk to protect recall. The top `RETRIEVE_K` (default 10) candidate pages are returned.
2. **Rerank** (`src/reranker.py`): the candidates are sent to Gemini as **downscaled thumbnails** (a cheap triage pass), and it returns the `RERANK_K` (default 2) pages that actually help answer the question. This keeps the answer step focused and sharpens the citation, without paying full-resolution image cost just to sort candidates. If the call fails or returns junk, it falls back to the top pages by MaxSim score.
3. **Answer** (`src/answerer.py`): the reranked **page images** are sent to Gemini at full resolution, which returns structured JSON: the `answer`, which `source_page` it came from, and a `box` in Gemini's native `[ymin, xmin, ymax, xmax]` convention normalized to a 0–1000 scale.
4. **Highlight** (`src/highlight.py`): the box is converted to pixels against the real page PNG (with a little padding), then **cropped** and **annotated**, saved under `page_images/crops/`.

The steps are wired as a small [LangGraph](https://langchain-ai.github.io/langgraph/) flow in `src/graph.py`: `retrieve → rerank → answer → highlight`.

## Project layout

```
src/
  config.py        # paths, model names, Qdrant + DPI + retrieve/rerank settings
  pdf_render.py    # PDF → page PNGs (pdf2image / Poppler)
  embedder.py      # ColQwen2 image + query embeddings
  vector_store.py  # Qdrant multivector store (upsert / search / delete, binary quantized)
  ingest.py        # ingest CLI: render → embed → batched upsert (incremental, or --rebuild)
  reranker.py      # Gemini thumbnail rerank: candidates → the pages that matter
  answerer.py      # Gemini structured answer + bounding box
  highlight.py     # crop + annotate the cited region
  graph.py         # LangGraph: retrieve → rerank → answer → highlight
  main.py          # query CLI (run_query seam + CLI wrapper)
  server.py        # warm FastAPI service: /query /health /corpus /ingest /heatmap + images
scripts/
  make_sample_pdf.py     # generates the text-layer-free sample PDF
  fetch_eval_corpus.py   # downloads + sha256-verifies the distractor corpus into pdfs/
eval/
  dataset.jsonl        # labeled questions: gold {pdf, page} + expected substrings
  corpus_manifest.json # the distractor corpus, pinned by sha256 (PDFs not committed)
  scoring.py           # pure scoring logic (recall@k, citation, abstention, coverage, calibration)
  run_eval.py          # eval CLI: retrieval-only / full / judge, JSON report + table
ui/                   # React + Vite UI: three-column workspace with visual citations
docker-compose.yml    # Qdrant vector database service
pdfs/                  # source PDFs to index
page_images/          # rendered pages + crops/ (generated, gitignored)
qdrant_data/          # embedded on-disk fallback store (generated, gitignored)
```

## Configuration

Knobs live in `src/config.py`:

| Setting | Default | Notes |
|---|---|---|
| `QDRANT_URL` | _(unset)_ | Qdrant server URL, e.g. `http://localhost:6333`; unset falls back to the embedded on-disk store. Set in `.env` |
| `COLPALI_MODEL` | `vidore/colqwen2-v1.0` | swap to `vidore/colqwen2.5-v0.2` for higher chart/table accuracy on a bigger GPU |
| `RENDER_DPI` | `150` | page render resolution |
| `RETRIEVE_K` | `10` | candidate pages pulled from Qdrant per query |
| `RERANK_K` | `2` | pages kept after the Gemini rerank, then sent to the answer step |
| `RERANK_THUMBNAIL_EDGE` | `768` | long-edge px for rerank thumbnails; set `None` to rerank on full-res pages |
| `GEMINI_MODEL` | `gemini-3.5-flash` | any vision-capable Gemini model (used for both rerank and answer) |
| `RERANK_MODEL` | _(= `GEMINI_MODEL`)_ | override to point the coarser rerank triage at a cheaper/faster model |
| `EVAL_JUDGE_MODEL` | _(= `GEMINI_MODEL`)_ | model the eval `--judge` flag grades answers with |

## Observability

Every query is traceable end to end. Set `LOG_JSON=true` for one JSON object per log
line (ready for a log aggregator); each line carries a per-query `request_id`, so the
`retrieve → rerank → answer → highlight` node timings (`latency_ms`), the per-call
Gemini token/cost lines, and a final `query complete` summary (total latency +
aggregated tokens/cost) all correlate. Human-readable lines are the default. A rerank
or answer step that fails degrades gracefully **and** logs a `degraded` warning, so a
silently-degraded query is still visible in the logs. The same per-query totals — plus a
**per-stage** (retrieve / rerank / answer / highlight) breakdown of time, tokens, and
cost — are also returned in the `/query` response's `meta` field, which the UI's *how
this was answered* trace renders.

**LangSmith tracing (optional).** Off by default and needs no code change. Set both
`LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` (optionally `LANGSMITH_PROJECT`) and
LangGraph emits traces natively. Each trace is tagged with the same `request_id` as
the logs, so a slow query in LangSmith maps straight back to its log lines.

| Setting | Default | Notes |
|---|---|---|
| `LOG_LEVEL` | `INFO` | stdlib log level |
| `LOG_JSON` | `false` | `true` emits one JSON object per line with `request_id` + `latency_ms` + token totals |
| `LANGSMITH_TRACING` | _(unset)_ | set `true` (with a key) to turn on LangSmith tracing |
| `LANGSMITH_API_KEY` | _(unset)_ | LangSmith API key; required for tracing |

## Development

The test suite is **pure logic** — it stubs the Gemini choke point (`gemini_client.generate`)
and image loaders, so no models, API key, network, or PNGs are touched (~seconds). It
covers the geometry, the vector-store alias logic, the observability plumbing, and the
FastAPI serving layer (via `TestClient` with the pipeline seam stubbed):

```bash
uv run pytest                     # backend
cd ui && npm run typecheck && npm run test   # UI: types + pure-logic units
```

**Lint & types** are enforced by `ruff` and `mypy` (in the `lint` dependency group):

```bash
uv sync --group lint              # install the tooling
uv run ruff check .               # lint (default rules + import sorting)
uv run mypy src eval              # type-check
```

**CI** (`.github/workflows/ci.yml`) runs on every push to `main` and every PR: a fast
`lint` job (ruff, no ML install) and a `test` job that installs the full stack and runs
`mypy` + the backend suite.

## Evaluation

A labeled set (`eval/dataset.jsonl`, 69 questions: 59 answerable with gold `{pdf, page}`
labels and expected-answer substrings, plus 10 unanswerable questions without gold pages)
plus a scoring harness make regressions visible: re-run after changing `RENDER_DPI`,
`RERANK_K`, or a model and diff the JSON reports to *prove* nothing regressed. Each
report carries a `config` snapshot so the two runs are comparable at a glance.

**The corpus is part of the instrument.** An earlier version of this eval scored 1.0
on recall@10, rerank recall, citation accuracy, substring match *and* the judge — not
because the pipeline was perfect but because the corpus was 43 pages, so a 10-candidate
retrieval returned 23% of the index and the gold page could not fail to be in it. Every
downstream metric inherited that ceiling, which made the harness useless as a guard: a
real regression had nowhere to show up. `scripts/fetch_eval_corpus.py` fixes it
structurally by adding 320 pages of deliberately confusable papers — the retrieval
family (ColBERT, ColBERTv2, DPR, BEIR, SPLADE, E5, RAG) carries the same
late-interaction prose and nDCG tables the ColPali questions turn on, and PaliGemma is
ColPali's own base model.

```bash
# Retrieval only — recall@k against the index, no Gemini calls (runs without a key)
GEMINI_API_KEY= PYTHONPATH=. uv run python eval/run_eval.py --retrieval-only

# Full pipeline — recall@k + rerank recall + citation correctness + substring match
PYTHONPATH=. uv run python eval/run_eval.py

# …plus LLM-as-judge scoring of each answer against the reference (EVAL_JUDGE_MODEL)
PYTHONPATH=. uv run python eval/run_eval.py --judge

# CI gate: exit 1 if any watched metric drops below its floor (repeatable, one run)
PYTHONPATH=. uv run python eval/run_eval.py --judge \
  --gate recall@1:0.70 --gate recall@3:0.88 --gate citation_accuracy:0.91 \
  --gate gold_coverage_avg:0.55 --gate abstention_accuracy:0.90
```

Those floors sit ~3 questions below the pinned baseline. Note what is *not* gated:
`recall@10` and `rerank_recall` are both still 1.0 and gating them would guard
nothing — see below.

Reports land in `eval/reports/` (gitignored, except the pinned
`baseline_desaturated.json`). Metrics:

| family | question it answers |
| --- | --- |
| **recall@k** | is the gold page in Qdrant's top-`RETRIEVE_K`, and in the reranked top-`RERANK_K`? |
| **citation accuracy** | did the answer's `source_page` resolve to the gold page? |
| **answer quality** | substring match, plus the optional LLM judge |
| **abstention accuracy** | on the 10 questions the corpus *cannot* answer, did it decline instead of inventing one? Its complement is the hallucination rate. |
| **gold-document coverage** | on the 6 questions whose gold spans two PDFs, did rerank spend a slot on each? The only metric that puts `RERANK_K` under pressure. |
| **confidence separation** | is confidence higher on correct citations than wrong ones? ~0 means the signal is noise. |

Each is also sliced by tag (`chart` / `table` / `figure` / `formula` / `text` /
`cross-doc` / `unanswerable`), and every metric is computed over *applicable rows
only* — an unanswerable question carries no gold page, so it scores abstention without
entering a recall denominator. The scoring logic (`eval/scoring.py`) is pure and
unit-tested; the full run reuses `main.run_query`, so it also reports per-question
latency/token/cost for free.

### Baseline

`eval/reports/baseline_desaturated.json` is committed as the thing to diff against.

| metric | 53 questions / 43 pages | 69 questions / 363 pages |
| --- | --- | --- |
| recall@1 | 0.8302 | 0.7627 |
| recall@3 | 0.9623 | 0.9322 |
| recall@10 | 1.0000 | 1.0000 |
| rerank_recall | 1.0000 | 1.0000 |
| citation_accuracy | 1.0000 | 1.0000 |
| substring_accuracy | 1.0000 | 1.0000 |
| judge_accuracy | 1.0000 | 0.9831 |
| **gold_coverage_avg** | — | **0.7500** |
| **abstention_accuracy** | — | **1.0000** (10/10) |
| **confidence_separation** | — | *n/a this run* |

**What the de-saturation actually found.** Not more retrieval headroom — on the
*identical* 53 questions the 8× bigger corpus moved recall@1/@3/@10 by exactly zero,
and `recall@10` and `rerank_recall` are still pinned at 1.0. What it found is that
`gold_doc_coverage` catches **answers that are correct but not grounded in the
retrieved pages**. On `xdoc-ndcg-cutoffs` the pipeline answered *"ColPali reports
nDCG@5… BEIR is scored using nDCG@10"* — right on both halves, scored correct by
citation, substring *and* the judge — while `beir.pdf` was never in the reranked set at
all. The BEIR half came from the model's parametric knowledge, not from a page it was
shown. For a system whose premise is *"here is the box where I read this,"* that is the
failure mode that matters, and `gold_coverage_avg` is the only metric that sees it.
Four of six cross-document questions score 0.5 for the same reason: `RERANK_K=2` spent
both slots inside one document.

**Read these numbers with their run-to-run variance.** Across four valid runs of
identical code, `citation_accuracy` was 0.9661 three times and 1.0000 once, and
`gold_coverage_avg` was 0.6667 three times and 0.7500 once. On denominators of 59 and 6
respectively, one question moves them by 0.017 and 0.083 — so a single row's movement is
noise, not a regression. `judge_accuracy` is the same story: `sales-q2-revenue`, where
the judge rejects `$150,000` for a chart labeled "Thousands" showing 150, has flipped
miss/pass/miss/pass across those runs. None of the three is gated tightly for that reason.

**`confidence_separation` has a design flaw worth knowing about.** It is the gap between
retrieval confidence on correct citations and on wrong ones, so it needs at least one
wrong citation to exist — and reports `n/a` when the pipeline gets everything right, as
it did in the pinned run. The metric vanishes exactly when things go well. The component
averages are still reported (`retrieval_conf_correct_avg` 0.1319 here), and the earlier
runs that did have wrong citations put the separation at ~0.032, i.e. close to noise.

Two smaller results worth knowing: the model self-reported `high` confidence on all 59
answerable questions (a degenerate signal), and the deterministic retrieval confidence
separates correct from wrong citations by only 0.032 — both are surfaced in the UI and
neither currently carries much information.

## Notes

- Qdrant runs as a **Dockerized server** (`docker compose up -d`) with **binary quantization** on the multivectors — 128-d → 128 bits in RAM, full-precision vectors on disk for rescoring — so the index scales to hundreds of pages. Leave `QDRANT_URL` unset to fall back to the **embedded on-disk** store (`qdrant_data/`) for quick prototyping with no container.
- The sample PDF is deliberately **pixel-only** (no selectable text) to prove the vision path does the work.
- Generated data (`qdrant_data/`, `page_images/`) is gitignored and rebuilt by ingest; the server's index lives in the `qdrant_storage` Docker volume.
