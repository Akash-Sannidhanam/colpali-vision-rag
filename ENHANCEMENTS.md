# Enhancements / Backlog

Possible improvements to the vision-citation pipeline. None are blocking: the
core feature (Gemini returns a bounding region, which is cropped and shown to
the reader) is complete, tested, and shipped. Captured here so they are not lost.

> **Operational roadmap:** the staged production-hardening pass (warm serving,
> reliability, observability, evaluation) is tracked in
> [PRODUCTION_HARDENING.md](PRODUCTION_HARDENING.md). Several items below feed
> into it — cross-referenced inline as _(→ Hardening Phase N)_.

## Display

- **Inline UI for the crop.** _(→ Hardening Phase 3.)_ Today the answer prints to the terminal and the
  crop opens in macOS Preview (`_open_file` in `src/main.py`). A small Streamlit
  or Gradio app would render the answer, the cropped slice, and the annotated
  page together in the browser, which is the natural home for a "show the reader
  the exact slice" feature. Highest-value next step.
- **Cross-platform auto-open.** _(→ Hardening Phase 3 — the inline UI supersedes this.)_ `_open_file` in `src/main.py` only handles macOS
  (`open`). Add `xdg-open` (Linux) and `os.startfile` / `start` (Windows), or
  drop auto-open entirely once an inline UI exists.

## Artifacts

- **Persist crops per query.** `crop_region` and `annotate_page` in
  `src/highlight.py` write `<page_stem>_crop.png` and overwrite on every run.
  Adding a short hash of the question to the filename keeps a history across
  queries instead of clobbering the previous one.

## Robustness

- **Harden structured-output parsing.** _(→ Hardening Phase 1.)_ `src/answerer.py` falls back to
  `json.loads(response.text)` when `response.parsed` is None; a malformed
  response would raise. Catch that and return a not-found result so the CLI
  degrades gracefully. `src/reranker.py` already wraps its Gemini call in a
  broad `try/except` that falls back to the Qdrant top-k — `answerer.py` should
  adopt the same pattern.

## Scope

- **Multiple regions.** The pipeline cites a single primary region (a deliberate
  choice). If an answer ever spans two pages or two areas, extend the `Citation`
  schema in `src/answerer.py` to a list of boxes and crop each.
- **Integration test with a mocked Gemini.** _(→ Hardening Phase 1 verify / Phase 4.)_ Current tests cover the pure
  geometry in `src/highlight.py` and the pure rank-cleaning logic
  (`_valid_order`) in `src/reranker.py`. A test that stubs the Gemini calls
  would additionally cover the `rerank_node → answer_node → highlight_node`
  wiring without needing an API key.

## Retrieval / rerank

- **Cheaper/faster rerank model.** _(→ `RERANK_MODEL` knob added in Hardening
  Phase 0; wiring into `reranker.py` lands in Phase 1.)_ `src/reranker.py` reuses
  `GEMINI_MODEL` for the triage pass. A lighter model (e.g. a Flash-Lite tier)
  would cut the rerank call's cost and latency further, since picking page indices
  is a coarser task than reading the answer — via the `RERANK_MODEL` knob.
- **Adaptive rerank count.** `RERANK_K` is a fixed 2, and `_valid_order` tops up
  to exactly `k`. Letting the reranker keep a variable number of pages (1 when a
  single page clearly answers, more when the answer spans pages) would trade a
  little predictability for precision.
- **Surface the dropped candidates.** `rerank_node` overwrites `retrieved`, so
  `main.py` prints only the kept pages. Adding a `candidates` key to `RAGState`
  would let the CLI show "retrieved 10, used 2" for transparency.
