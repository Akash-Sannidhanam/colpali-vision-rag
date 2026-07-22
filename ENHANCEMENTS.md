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

- **Harden structured-output parsing.** _(✅ done in Hardening Phase 1.)_
  `src/answerer.py` now routes through `gemini_client.generate` inside a
  `try/except` (mirroring `reranker.py`) and returns a well-formed not-found
  citation on any failure — including a malformed or wrong-shape response, which
  is re-validated through the `Citation` model. `highlight_node` skips it cleanly.

## Scope

- **Multiple regions.** The pipeline cites a single primary region (a deliberate
  choice). If an answer ever spans two pages or two areas, extend the `Citation`
  schema in `src/answerer.py` to a list of boxes and crop each.
- **Integration test with a mocked Gemini.** _(✅ done in Hardening Phase 4.)_
  `tests/test_answerer.py` covers the answer/highlight wiring, and
  `tests/test_pipeline_integration.py` now exercises the whole compiled graph
  (retrieve→rerank→answer→highlight) with every boundary stubbed: rerank-order
  alignment into the highlight, Qdrant-top-k fallback, garbage-index cleanup, answer
  degradation, malformed JSON, and empty retrieval — no API key or PNGs.

## Retrieval / rerank

- **Cheaper/faster rerank model.** _(✅ `RERANK_MODEL` wired into `reranker.py` in
  Hardening Phase 1.)_ The rerank triage now calls `gemini_client.generate` with
  `RERANK_MODEL` (defaults to `GEMINI_MODEL`). Point it at a lighter tier (e.g.
  Flash-Lite) via `.env` to cut the rerank call's cost/latency — picking page
  indices is a coarser task than reading the answer.
- **Adaptive rerank count.** _(✗ evaluated & rejected in the retrieval-quality pass.)_
  `RERANK_ADAPTIVE` (config, default off) makes `_valid_order(top_up=False)` keep a
  variable 1..`RERANK_K` pages instead of padding to `k`. On the hardened 53-question
  eval it showed no citation/precision gain (citation_accuracy stays 1.0) and a small
  judge-score dip, for ~5% lower latency — not worth flipping on for this corpus. Kept
  as a knob for larger corpora where the extra page distracts the answer step.
- **Surface the dropped candidates.** _(◐ half-done in Hardening Phase 4.)_
  `retrieve_node` now writes the untrimmed top-k to a `candidates` `RAGState` key
  (added so the eval harness can score recall@k), so the data is already threaded
  through `run_query`'s result and the `/query` response. What's left is purely
  presentational: have the CLI / UI show "retrieved 10, used 2" from it.
