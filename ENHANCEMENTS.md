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

## Ingest throughput

- **The lever was the visual-token budget, not batching.** _(✅ done — see the
  ingest-throughput pass in PRODUCTION_HARDENING.md.)_ The batch-embedder pass measured
  *which stage* was slow (embed, 98%) but never measured *inside* it, and that changed the
  conclusion. Splitting embed into preprocess / forward / decode showed the forward pass is
  ~5.45 s of a 6.24 s page — and it is slow because the model sees **755 visual tokens**,
  not because pages go one at a time. The checkpoint caps at 768 (`max_pixels=602112`) and
  `ColQwen2Processor.from_pretrained` takes a `max_num_visual_tokens` kwarg the code never
  passed. ViT attention is quadratic in patch count, so the budget pays superlinearly:
  512 tokens is **2.04×**, 256 is **3.95×**. Now `EMBED_VISUAL_TOKENS`, in `EMBED_VERSION`
  (it changes every vector — the exact inverse of `EMBED_BATCH_SIZE`, which stays out
  because batching is vector-identical).
- **Two levers that look obvious and do nothing**, recorded so they are not retried.
  Lowering `RENDER_DPI` cannot speed up embedding — `smart_resize` downscales a 2.1 MP page
  to 672×868 before the model sees it, so the model already reads pages at ~79 DPI
  equivalent; DPI buys render time and disk, and the page PNGs still need it for the
  full-res answer step. And fp16 vs bf16 measured 6.5 s vs 6.5 s, inside the noise floor.
- **MRL is the wrong tool here, twice over.** _(investigated, not scoped.)_ ColQwen2-v1.0 is
  a LoRA adapter over `vidore/colqwen2-base` with a fixed-width `custom_text_proj` and no
  matryoshka config, so truncating dimensions would degrade unpredictably rather than
  gracefully. More fundamentally it targets the 1536→128 projection, not the 2.25B backbone,
  so it cannot move the forward pass at all — it is a storage lever, and
  `vector_store.ensure_collection` already pulls a harder one (`BinaryQuantization` at 32×
  against MRL's 2–4×).
- **The upsert was on the critical path and had never been measured.** _(✅ fixed.)_
  `upsert_pages` and `save_page_image` ran inline in the embed loop, and the profiler's
  `--store` was opt-in, so `bench/reports/ingest_baseline.json` carried
  `upsert_measured: false` through an entire ingest-optimisation pass. Measured: 0.264 s/page.
  It is measured by default now, and both it and the preprocess run off the GPU's thread.
- **Getting the CPU off the GPU's path is worth less than its serial cost suggests.**
  _(measured, and the reason is the point.)_ GPU idle went 9.7% → 7.0%, not the ~13% the
  stage costs predicted, because background Python threads contend with the GPU dispatch
  thread for the GIL. Attributed: the preprocess lookahead won 3 of 3 interleaved rounds,
  the store worker **0 of 3** — its own GIL traffic costs the main thread about what moving
  the work off it saves. The store worker is kept only because its share grows as the
  forward pass shrinks; re-measure and delete it if it stays flat.
- **Escaping the GIL for preprocess.** _(open.)_ A process pool rather than a thread pool
  would remove the contention above, at the cost of pickling ~5 MB of `pixel_values` per
  batch across the boundary. Worth trying only if a profile shows preprocess still inflating
  the measured forward time after the token budget lands.

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
  _(✅ swept — and the gap was much smaller than this note assumed. See the
  cross-document attribution pass in PRODUCTION_HARDENING.md.)_ All 20 cross-document
  rows were swept with `scripts/find_in_pdfs.py`; 11 gained pages. `docvqa.pdf` p.3 is
  real and now labelled, as are `beir.pdf` p.7/p.9. But re-scoring the stored retrieval
  candidates against the new labels moved coverage on **exactly one row** (0.675 →
  0.700). The pinned figure was a lower bound by ~0.025, not by the wide margin the
  original note feared. The sweep also fixed a methodological trap worth remembering:
  `pdftotext` matches **running headers**, and "OCR-free Document Understanding
  Transformer" is `donut.pdf`'s title on every odd page. Counting page furniture would
  have made half that document gold and turned document-level coverage into a tautology.
  The original note follows.

  Coverage did not
  move at all on the winning arm, because the extra rerank slot pulled in pages that
  state the fact but are not in the row's gold list — `docvqa.pdf` p.3 says "The DocVQA
  comprises 50,000 questions" and is unlabelled; `beir.pdf` states its 18 datasets on p.7
  and p.9 beyond the labelled p.1–3. Every pinned `gold_coverage_avg` is a lower bound
  until the 20 cross-document rows are swept for the same gap. The sweep is mechanical —
  `scripts/find_in_pdfs.py` with the fact's regex, then add every page that states it —
  but it invalidates the pinned baseline, so it wants its own pass with a re-run. Worth
  doing before `gold_doc_coverage` is trusted to adjudicate anything else.

- **One document monopolizes the candidate slate on cross-document questions.**
  _(✅ fixed — `MAX_PAGES_PER_DOC=5` with `RETRIEVE_K=12` takes `candidate_coverage_avg`
  0.700 → 0.825 with no gold page evicted, and the judged run carried `gold_coverage_avg`
  the full distance with it, 0.675 → 0.825. The two are now **equal**: rerank loses nothing,
  and all remaining coverage headroom is retrieval-side. Cost: latency +20% and one net
  citation question. See the slate-diversity pass in PRODUCTION_HARDENING.md.)_
  Two corrections to the note below, both from the arms. **Widening `RETRIEVE_K` 10 → 12 is
  worth 0.100 of the 0.125 on its own** — the boring lever, never tried, because the
  attribution pass framed the problem as diversity and went looking for a diversity fix. And
  **the predicted cap of 4 is the wrong value**: it scores higher coverage (0.850) but evicts
  `colpali.pdf` p7 — a *single*-document question's gold, the 5th colpali page in the ranking —
  and backfills with six unrelated documents. A cap only pays where the question spans
  documents, and 63 of the 83 questions do not. The original note follows.

  With the attribution pair
  (`candidate_doc_coverage` vs `gold_doc_coverage`) in place, **11 of the 12 coverage
  misses are retrieval-side**: the reranker preserved coverage on every row where
  retrieval offered both documents, losing exactly one. Seven of those misses are the
  second document being **entirely absent from the top-10**, and the mechanism is
  starker than "it ranked 11th" — on three rows the top-10 is 10 pages of a *single*
  PDF (`colpali.pdf` ×10 shuts out `paligemma.pdf`; `albert.pdf` ×10 shuts out
  `bert.pdf`; `rag.pdf` ×10 shuts out `dpr.pdf`), and on a fourth it is 9 of 10. ColQwen2's
  MaxSim on a two-part query is dominated by whichever document matches more of the query
  tokens, and it takes every slot.
  This makes `candidate_coverage_avg` a **hard ceiling** on `gold_coverage_avg`: no
  rerank change, and no `RERANK_K`, can lift coverage past what retrieval offered.
  **The `RETRIEVE_K=50` probe settled it** (free, deterministic): `candidate_coverage_avg`
  is **0.975 at k=50 against 0.700 at k=10**, so retrieval reaches both documents on 19.5
  of 20 rows once the slate is deep enough — the ceiling is not ColQwen2's ranking, it is
  slate diversity. And 6 of the 7 shut-out documents' gold pages sit at rank **1–4 within
  their own document** (global 11–41): beir 11/#1, paligemma 15/#1, siglip 15/#2, bert
  16/#3, dpr 20/#4, dpr 41/#4. So a **per-document cap of 4** on the 10-slot slate recovers
  six of seven without deepening retrieval — via a cap in `vector_store.search` or Qdrant
  `query_points_groups(group_by="pdf")`. Only `xdoc-splade-vocab-dpr-dense`'s `dpr.pdf`
  page is outside the top-50 entirely and needs query decomposition (embed each half of a
  two-part question, union the candidates). Guard the change with the new
  `candidate_coverage_avg` gate — it is deterministic, so a diversity win shows up with no
  LLM variance and needs no judged run to see.
- **Query decomposition for the one row that no slate policy can reach.** _(✅ done — the
  prescription in the note above was right. `QUERY_DECOMPOSE=true` +
  `DECOMPOSE_ORIGINAL_WEIGHT=0` took `candidate_coverage_avg` 0.825 → **0.850**,
  `rerank_recall` to **1.0**, citation 0.9315 → **0.9589** and judge 0.9178 → **0.9452**,
  and fixed `xdoc-splade-vocab-dpr-dense`. See the query-decomposition pass in
  PRODUCTION_HARDENING.md.)_ Three things the note above could not have known.
  **"Union the candidates" is underspecified and two of its three readings lose.** Fusing
  by score hands the slate to whichever half is wordier, because MaxSim sums over query
  tokens. Fusing by rank while *also* including the whole question is worse than it sounds:
  RRF rewards agreement, so the query that cannot find the second document out-votes the
  half that can — that arm bought **zero** coverage for 9 rows of worsened gold rank.
  Only halves-only RRF works.
  **The recall proxies mispriced the change.** recall@1 0.7397 → 0.6712 and recall@3
  0.9041 → 0.8493, while every answer-level metric improved — `RERANK_K=3` picks from a
  12-page slate, so what gates the answer is whether gold is *in* it, not where. Both
  floors had to be lowered; that is the honest cost of adopting.
  **`gold_coverage_avg` did not follow its ceiling** (0.825 against 0.850), so the slate
  pass's "rerank loses nothing" property no longer holds. That is now the sharpest open lead.
- **The confidence signals: the verdict below was itself unmeasured, and it was wrong.**
  _(✅ done — see the confidence-calibration pass in PRODUCTION_HARDENING.md.)_ Three
  corrections, in ascending order of how much they cost to learn.
  **`confidence_separation` was reported to four decimals off one row.** The pinned
  `baseline_decomposed.json` has 70 correct citations and **exactly one** wrong one
  carrying a confidence value, so its `-0.0062` is a single data point; `baseline_diverse`
  says `+0.0193` (n=3) and `baseline_swept` `+0.0226` (n=2). Same quantity, opposite sign,
  all noise. The metric degrades precisely as the pipeline improves, because its negative
  class *is* the pipeline's mistakes. `MIN_CALIBRATION_N=5` now withholds every calibration
  comparison below the floor, and every one ships with its counts.
  **The right label was free and 24× larger.** `src/confidence.py` measures retrieval
  decisiveness, and pairing it with `gold_rank` instead of `citation_correct` swaps a
  negative class of 1 for one of 24 — deterministic, no API key, no judged run. Measured
  there, the signal **does** carry information: AUC **0.629**, permutation p **0.016** at
  n=49/24. Weak, but not the "almost none" claimed below.
  **The defect was the presentation, not the formula.** `retrieval_confidence` is a
  softmax share over `RETRIEVE_K` candidates, so its floor is `1/12 = 8.3%`, not zero, and
  its entire observed range across 83 questions is 0.063–0.212. The UI rendered it as
  `Math.round(x*100)`, so a maximally decisive retrieval read **"21%"** and a typical
  correct answer read **"11%"** — the system appearing to doubt answers it got right. It
  now shows as a multiple of an evenly-spread slate (`1.50× uniform`) inside the trace
  disclosure rather than as a chip beside the answer, which is the placement AUC 0.629
  supports. The original note follows.

  The model
  self-reported `high` on most of the 73 answerable questions, showing some variance
  (high-confidence accuracy: 0.958, low-confidence: 0.0), but the deterministic
  retrieval confidence separates correct from wrong citations by only 0.0247. Both are
  surfaced in the UI. Either calibrate them or stop showing them as if they mean
  something; the eval now reports `confidence_separation` to tell.

- **`self_conf_low_acc` is a tautology, and read as calibration for two passes.**
  _(✅ suppressed by the same floor.)_ `answerer._normalize` pins `confidence = "low"`
  onto every not-found answer, and a declined *answerable* row scores
  `citation_correct=False` — so every `low` row is a decline and every decline is scored
  wrong. The bucket is forced to 0.0 by the code whenever any decline exists, which is
  what "low-confidence accuracy: 0.0" above actually reported, from two rows. The
  invariant is correct and stays; the floor is what stops it being read as a measurement
  of the model's calibration.
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
