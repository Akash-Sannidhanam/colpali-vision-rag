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

# 2. Ingest: render pages → embed with ColQwen2 → store in Qdrant
PYTHONPATH=. uv run python src/ingest.py            # indexes everything in pdfs/
#   or point at specific files:  ... src/ingest.py path/to/doc.pdf

# 3. Ask a question
PYTHONPATH=. uv run python src/main.py "What was the Q4 revenue in the chart?"
```

The repo ships a small starter corpus in `pdfs/` — the generated sales report plus
two arXiv papers (*Attention Is All You Need* and *ColPali*, ~43 pages total) — so
the rerank step has a real 10-candidate pool to narrow out of the box. Drop your own
PDFs into `pdfs/` and re-run step 2 to index them too.

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
| `POST /ingest` (multipart PDF) | Render → embed → index an uploaded PDF. Blocking and long-running — it holds the model lock for the whole build. |
| `GET /images/...` | Static page / crop / annotated PNGs. |

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What was the Q4 revenue in the chart?"}' | jq .answer   # "180"
```

| Setting | Default | Notes |
|---|---|---|
| `SERVER_HOST` | `127.0.0.1` | uvicorn bind host (the `python src/server.py` runner) |
| `SERVER_PORT` | `8000` | uvicorn bind port |
| `CORS_ALLOW_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | comma-separated browser origins allowed to call the API; `*` allows any |
| `MAX_UPLOAD_MB` | `50` | reject larger PDF uploads to `POST /ingest` |

### UI (`ui/`)

A React + Vite single-page app — a three-column workspace (corpus rail · conversation ·
document viewer) that renders each answer with its **visual citation**: the cited page
with the bounding box drawn over it, the cropped slice, and the reranked-candidate rail,
plus a "how this was answered" per-stage trace. A **"why this page?"** toggle on the viewer
overlays the MaxSim patch heatmap (via `POST /heatmap`), tinting the patches the query
matched — the retrieval-side complement to the answer crop. It runs as its own dev server
and calls the API above (two processes: the API on `:8000`, the UI on `:5173`).

```bash
cd ui
npm install
npm run dev            # http://localhost:5173  (expects the API on :8000)
```

Point it at a non-default API with `VITE_API_BASE` (e.g.
`VITE_API_BASE=http://host:8000 npm run dev`). The API already allows the Vite dev
origin via CORS. `npm run typecheck` and `npm run test` cover the UI's pure logic (the
`citation.box → overlay` math and 1-based page resolution).

### Deployment (Docker)

The `Dockerfile` packages the **backend API** (the UI is served separately). It's a
multi-stage `uv` build on a slim base; on Linux it pulls the CUDA 12.8 torch wheels, so
the image is GPU-capable with `--gpus all` and **auto-falls back to CPU** when no GPU is
present. Poppler is included; it runs as a non-root user and serves on `0.0.0.0:8000`.

```bash
docker build -t vision-rag .
docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -v vision-rag-hf:/home/appuser/.cache/huggingface \   # persist the model download
  vision-rag                                            # add --gpus all on a GPU host
```

First boot downloads the ~2B ColQwen2 model into the mounted HF cache (subsequent boots
are warm); `/health` returns `503` until the model is loaded and Qdrant is reachable.
Qdrant must be reachable at `QDRANT_URL` — the model isn't baked in, and Qdrant runs
separately (see the compose file).

Or bring up the whole stack with Compose — the `app` service is wired to the `qdrant`
service and passes `GEMINI_API_KEY` through from your environment:

```bash
GEMINI_API_KEY=... docker compose up          # Qdrant + the containerized API
docker compose up -d qdrant                    # Qdrant only (run the app on the host)
```

On a GPU host, uncomment the `deploy.resources` block in the `app` service (needs the
NVIDIA container toolkit).

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
  vector_store.py  # Qdrant multivector store (create / upsert / search, binary quantized)
  ingest.py        # ingest CLI: render → embed → batched upsert
  reranker.py      # Gemini thumbnail rerank: candidates → the pages that matter
  answerer.py      # Gemini structured answer + bounding box
  highlight.py     # crop + annotate the cited region
  graph.py         # LangGraph: retrieve → rerank → answer → highlight
  main.py          # query CLI (run_query seam + CLI wrapper)
  server.py        # warm FastAPI service: /query /health /corpus /ingest + static images
scripts/
  make_sample_pdf.py   # generates the text-layer-free sample PDF
eval/
  dataset.jsonl        # labeled questions: gold {pdf, page} + expected substrings
  scoring.py           # pure scoring logic (recall@k, citation, substring, aggregate)
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

A small labeled set (`eval/dataset.jsonl`, ~22 questions over the sample corpus with
gold `{pdf, page}` labels and expected-answer substrings) plus a scoring harness make
regressions visible: re-run after changing `RENDER_DPI`, `RERANK_K`, or a model and
diff the JSON reports to *prove* nothing regressed. Each report carries a `config`
snapshot so the two runs are comparable at a glance.

```bash
# Retrieval only — recall@k against the index, no Gemini calls (runs without a key)
GEMINI_API_KEY= PYTHONPATH=. uv run python eval/run_eval.py --retrieval-only

# Full pipeline — recall@k + rerank recall + citation correctness + substring match
PYTHONPATH=. uv run python eval/run_eval.py

# …plus LLM-as-judge scoring of each answer against the reference (EVAL_JUDGE_MODEL)
PYTHONPATH=. uv run python eval/run_eval.py --judge

# CI gate: exit 1 if retrieval recall@RETRIEVE_K drops below a threshold
PYTHONPATH=. uv run python eval/run_eval.py --retrieval-only --fail-under-recall 0.9
```

Reports land in `eval/reports/` (gitignored). Metrics: **recall@k** (is the gold page
in Qdrant's top-`RETRIEVE_K`, and within the reranked top-`RERANK_K`?), **citation
correctness** (did the answer's `source_page` resolve to the gold page?), **answer
quality** (substring match, plus the optional judge), each also sliced by tag
(`chart` / `table` / `figure` / `formula` / `text`). The scoring logic
(`eval/scoring.py`) is pure and unit-tested; the full run reuses `main.run_query`, so
it also reports per-question latency/token/cost for free.

## Notes

- Qdrant runs as a **Dockerized server** (`docker compose up -d`) with **binary quantization** on the multivectors — 128-d → 128 bits in RAM, full-precision vectors on disk for rescoring — so the index scales to hundreds of pages. Leave `QDRANT_URL` unset to fall back to the **embedded on-disk** store (`qdrant_data/`) for quick prototyping with no container.
- The sample PDF is deliberately **pixel-only** (no selectable text) to prove the vision path does the work.
- Generated data (`qdrant_data/`, `page_images/`) is gitignored and rebuilt by ingest; the server's index lives in the `qdrant_storage` Docker volume.
