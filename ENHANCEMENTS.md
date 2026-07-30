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

## Corpus lifecycle

- **Incremental ingest + document delete.** _(✅ done.)_ Adding a PDF used to
  re-render and re-embed the *entire* corpus (`POST /ingest` handed every file in
  `pdfs/` to a fresh-collection rebuild), and there was no way to remove a document
  at all. `run_ingest` now defaults to an incremental sync that upserts into the live
  collection and embeds only what changed — decided by a `content_hash` (sha256 of
  the PDF bytes) plus an `embed_version` (`COLPALI_MODEL@RENDER_DPI`) stored in every
  point's payload, so a model/DPI change re-embeds even when the bytes are identical.
  Point ids are `uuid5(pdf, page)`, so a re-ingest overwrites in place. `--rebuild`
  keeps the atomic wholesale path. `DELETE /corpus/{pdf}` (+ a remove action in the
  corpus rail) drops a document's vectors, page images, crops, and source PDF.

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
- **De-saturate the eval.** _(✅ done — see PRODUCTION_HARDENING.md.)_ The harness
  scored 1.0 on five metric families at once and so could not fail. Fixed by adding
  abstention, cross-document coverage and confidence-calibration scoring, 16 new
  questions (53 → 69), and a 320-page pinned distractor corpus. Two results worth
  carrying forward: growing the corpus 8× moved retrieval recall by **exactly zero** on
  the unchanged questions, so corpus size was the wrong lever — new *question types*
  were the cheap one; and `gold_coverage_avg` (0.667) catches answers that are correct
  but ungrounded in the retrieved pages, which no other metric here can see.
- **Raise `RERANK_K` for multi-document questions.** _(✅ done — `RERANK_K` defaults to 3.
  See the RERANK_K decision in PRODUCTION_HARDENING.md.)_ Three arms over 83 questions:
  fixed k=3 gave two genuine answer fixes plus a citation fix with **zero regressions
  across 144 paired row comparisons**, for +7.8% cost and flat median latency. Adaptive
  (`RERANK_ADAPTIVE=true`, cap 3) was cheaper and the only arm to move `rerank_recall`,
  but gave up the citation fix and had *worse* median latency; kept as a knob, not the
  default. The failure being fixed was sharper than expected — starved of the second
  document the model did not decline, it answered from memory with numbers that were
  plausible and wrong ("108M" for 110M, "19 datasets" for 18).

- **Gold labels are page-level and under-complete, so `gold_doc_coverage` under-counts.**
  _(new, and it cost the RERANK_K decision its pre-registered metric.)_ Coverage did not
  move at all on the winning arm, because the extra rerank slot pulled in pages that
  state the fact but are not in the row's gold list — `docvqa.pdf` p.3 says "The DocVQA
  comprises 50,000 questions" and is unlabelled; `beir.pdf` states its 18 datasets on p.7
  and p.9 beyond the labelled p.1–3. Every pinned `gold_coverage_avg` is a lower bound
  until the 20 cross-document rows are swept for the same gap. The sweep is mechanical —
  `scripts/find_in_pdfs.py` with the fact's regex, then add every page that states it —
  but it invalidates the pinned baseline, so it wants its own pass with a re-run. Worth
  doing before `gold_doc_coverage` is trusted to adjudicate anything else.
- **The confidence signals carry almost no information.** _(new, measured.)_ The model
  self-reported `high` on most of the 73 answerable questions, showing some variance
  (high-confidence accuracy: 0.958, low-confidence: 0.0), but the deterministic
  retrieval confidence separates correct from wrong citations by only 0.0247. Both are
  surfaced in the UI. Either calibrate them or stop showing them as if they mean
  something; the eval now reports `confidence_separation` to tell.
- **Refuse to pin a baseline from a degraded run.** _(✅ done — see the
  instrument-sharpening pass in PRODUCTION_HARDENING.md.)_ Built as described below, with
  one change: the report is still written, stamped `degraded_run` and named
  `degraded_<utc>.json` instead of `eval_<utc>.json`, rather than suppressed. Keeping the
  artifact is worth more for diagnosing the outage than the marginal safety of deleting
  it, and the filename plus the stamp are what actually prevent the mistake. Gates are
  skipped on such a run, not evaluated. The original note follows.

  A
  full `--judge` run against a depleted Gemini quota produced `citation_accuracy` 0.0,
  `substring_accuracy` 0.0 and every judge N/A — the graceful-degradation contract
  working exactly as designed — and still wrote a report that could have been committed
  as the reference. Worse, **`abstention_accuracy` scored 1.0 on that run**, because
  `abstention_correct` is `not citation["found"]` and cannot tell "correctly declined"
  from "never reached the model". The metric most dangerous to falsely peg at 1.0 is the
  one a total outage flatters most.
  `request_context` already records per-stage call counts and `gemini_client` already
  logs `degraded: True`, so the fix is to thread a degraded-call count into the report
  and have `run_eval` exit 2 — the existing setup-error code — when it exceeds a small
  fraction of rows, rather than writing a report at all. Cheap, pure, and testable
  without an API key.
