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

# add your key
echo 'GEMINI_API_KEY=your_key_here' > .env
```

`.env` is gitignored, so your key stays local.

## Usage

```bash
# 1. Generate the sample PDF (a bar chart + a sales table, pure pixels, no text layer)
uv run python scripts/make_sample_pdf.py

# 2. Ingest: render pages → embed with ColQwen2 → store in local Qdrant
PYTHONPATH=. uv run python src/ingest.py            # indexes everything in pdfs/
#   or point at specific files:  ... src/ingest.py path/to/doc.pdf

# 3. Ask a question
PYTHONPATH=. uv run python src/main.py "What was the Q4 revenue in the chart?"
```

Drop your own PDFs into `pdfs/` and re-run step 2 to index them.

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

`RETRIEVED PAGES` lists the pages kept *after* reranking. With a corpus larger than the two sample pages, retrieval pulls 10 candidates and the rerank step narrows them to the 2 shown here before the answer step runs.

## How it works

1. **Retrieve** (`src/embedder.py`, `src/vector_store.py`): the query is embedded into ColQwen2's token-level multivectors and matched against per-page multivectors in a local, on-disk Qdrant collection ranked by **MaxSim**. The top `RETRIEVE_K` (default 10) candidate pages are returned.
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
  vector_store.py  # local Qdrant multivector store (create / upsert / search)
  ingest.py        # ingest CLI: render → embed → upsert
  reranker.py      # Gemini thumbnail rerank: candidates → the pages that matter
  answerer.py      # Gemini structured answer + bounding box
  highlight.py     # crop + annotate the cited region
  graph.py         # LangGraph: retrieve → rerank → answer → highlight
  main.py          # query CLI
scripts/
  make_sample_pdf.py   # generates the text-layer-free sample PDF
pdfs/                  # source PDFs to index
page_images/          # rendered pages + crops/ (generated, gitignored)
qdrant_data/          # local Qdrant store (generated, gitignored)
```

## Configuration

Knobs live in `src/config.py`:

| Setting | Default | Notes |
|---|---|---|
| `COLPALI_MODEL` | `vidore/colqwen2-v1.0` | swap to `vidore/colqwen2.5-v0.2` for higher chart/table accuracy on a bigger GPU |
| `RENDER_DPI` | `150` | page render resolution |
| `RETRIEVE_K` | `10` | candidate pages pulled from Qdrant per query |
| `RERANK_K` | `2` | pages kept after the Gemini rerank, then sent to the answer step |
| `RERANK_THUMBNAIL_EDGE` | `768` | long-edge px for rerank thumbnails; set `None` to rerank on full-res pages |
| `GEMINI_MODEL` | `gemini-3.5-flash` | any vision-capable Gemini model (used for both rerank and answer) |

## Development

Run the geometry tests (pure Pillow, no models or API key required):

```bash
uv run pytest
```

## Notes

- Qdrant here is **embedded/on-disk** (`qdrant_data/`), not a server, so there are no extra services to run.
- The sample PDF is deliberately **pixel-only** (no selectable text) to prove the vision path does the work.
- Generated data (`qdrant_data/`, `page_images/`) is gitignored and rebuilt by ingest.
