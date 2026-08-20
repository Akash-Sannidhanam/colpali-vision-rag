# Experiments

Every retrieval and ingest decision in this pipeline was measured before it shipped. This
page is the record — **including the arms that were rejected**, which are the useful half.

Three things about the method are worth stating up front, because they are what make the
numbers here mean anything:

- **Rejected arms are written down with their numbers**, not deleted. A knob that looked
  obvious and did nothing is a result.
- **Hold-outs were built before the arms ran**, and deliberately not tuned against
  afterwards — that would convert the hold-out into training data.
- **Slate policy was simulated offline against a stored probe before any code was written**,
  and every live arm then reproduced the simulation to four decimal places.

The full working log, with the reasoning as it happened, is in
[ENGINEERING_LOG.md](ENGINEERING_LOG.md). The harness itself is `eval/run_eval.py`; see the
[Evaluation](../README.md#evaluation) section of the README for how to run it.

---

## How arms are run

**Retrieval-only arms are free and deterministic.** `--retrieval-only` needs no API key and
makes no Gemini calls, so a retrieval knob costs nothing to test and re-running an arm
reproduces it digit for digit. Only latency and answer-quality claims need the ~25-minute
judged run.

**An experiment arm is a prefix, not a code edit.** `RETRIEVE_K`, `RERANK_K`,
`MAX_PAGES_PER_DOC`, `CANDIDATE_FANOUT`, `QUERY_DECOMPOSE`, `MAX_SUBQUERIES` and
`DECOMPOSE_ORIGINAL_WEIGHT` are env-overridable and all seven land in the report's config
snapshot, so `diff_reports.py` names what changed between two runs.

**A stored report is a simulator.** Every answerable row stores the `[{pdf, page, score}, …]`
it retrieved and reranked, and `gold_rank` / both coverage metrics are pure functions of
(pages, gold). So a gold-label fix is re-scorable offline instead of costing a judged re-run,
and — since the scores are stored too — a candidate confidence formula can be scored against
a stored report for free rather than costing a live arm per idea.

**Flipped rows are audited individually before being counted.** Under-labeled gold has caused
more apparent regressions here than the pipeline has. `diff_reports.py` joins two reports on
row id and prints exactly which questions moved.

**A degraded run can never become the baseline.** A run against depleted Gemini quota still
produces a full report — and scores `abstention_accuracy` **1.0**, because a call that never
reached the model is indistinguishable from a correct refusal. Degraded calls are counted;
above 2% the run is stamped `degraded_run`, skips the gates, and exits 2.

---

## Retrieval

### Slate diversity ✓ ADOPTED

**Question:** three baseline rows returned *ten pages of a single PDF*, shutting the second
gold document out of the slate entirely. Can a per-document cap recover them?

**Method first:** the fix was scored before it was written. Re-scoring the stored
`RETRIEVE_K=50` probe offline settled four things at zero cost — cap=4 is optimal at k=10,
a 2× fanout pool saturates coverage, round-robin interleave is far worse (0.575), and
demote-instead-of-drop is *exactly* equivalent.

| arm | recall@1 | gold in slate | `cand_cov` |
|---|---|---|---|
| control, k=10 | 0.7397 | 0.9863 | 0.700 |
| cap=4, k=10 | 0.7397 | 0.9726 | 0.825 |
| k=12, no cap | 0.7397 | 0.9863 | 0.800 |
| **cap=5, k=12** | 0.7397 | **0.9863** | **0.825** |
| cap=4, k=12 | 0.7397 | 0.9726 | 0.850 |

**Result:** `gold_coverage_avg` 0.675 → **0.825**, moving the full distance to its ceiling —
rerank was then losing nothing at all. Cost: latency +20% (15.0 s → 18.0 s) to triage 12
thumbnails instead of 10.

**The finding that mattered wasn't the cap.** `RETRIEVE_K` 10 → 12 *alone* was worth 0.100 of
the 0.125, at zero recall cost — and it had never been tried, because the previous pass framed
the problem as diversity and went looking for a diversity fix. When a probe rules a mechanism
in, it does not rule out the boring lever that addresses the same symptom.

**The honest cost:** every cap=4 arm evicts one specific gold page
(`colpali-avg-ndcg`, the 5th colpali page in its own ranking). On a single-document question
diversity is pure loss, and 63 of 83 questions are single-document. cap=5 is the only setting
that takes the full win with nothing evicted.

### Query decomposition ✓ ADOPTED

**Question:** one row's gold page (`dpr.pdf` p3) sat outside the whole question's top-**50** —
no slate policy reaches it. Does splitting a two-part question and fusing the rankings?

| arm | recall@1 | recall@3 | recall@12 | `cand_cov` |
|---|---|---|---|---|
| control (off) | 0.7397 | 0.9041 | 0.9863 | 0.825 |
| whole + halves (weight 1) | 0.6986 | 0.8630 | 0.9863 | 0.825 |
| **halves only (weight 0)** | 0.6712 | 0.8493 | **1.0000** | **0.850** |

**Result:** adopted. The target row was fixed (coverage 0.5 → 1.0, now citing the gold page),
`rerank_recall` reached 1.0, and every answer-level metric improved — citation 0.9315 →
**0.9589**, substring 0.9444 → **0.9583**, judge 0.9178 → **0.9452**.

**Retrieval-precision proxies priced a cost the pipeline does not pay.** recall@1 and recall@3
both *fell* — the halves order the top of the slate worse — and the answers got better anyway,
because `RERANK_K=3` picks from a 12-page slate, so what gates the answer is whether gold is
**in** the slate at all. Adopting this lowered two gate floors; that is the real price.

**Fusing the whole question beside its halves actively suppresses the fix ✗.** RRF rewards
*agreement*, so including the whole question gives the query that **cannot** find the second
document an equal vote. The equal-weight arm bought **zero** coverage for 9 rows of worsened
gold rank. The design note predicting that keeping it was "strictly additive, therefore safer"
was wrong, and only the arm could show it.

**The hold-out earned its keep.** Every cross-document row in `dataset.jsonl` is phrased
`"<A>, and <B>?"`, so a splitter keyed on that string would score beautifully and prove only
that it memorised the dataset. `dataset_paraphrase.jsonl` re-phrases 12 rows across five forms.
The splitter fires on **5 of 12** — measured *before* the arms, and the splitter was then
deliberately **not** edited. On those 5, coverage moved 0.7917 → 0.8333, same direction and
magnitude as the main slice, on phrasings it had never been shown. That reach (~40% of natural
phrasings), not the coverage number, is the bar any replacement splitter has to beat.

### `RERANK_K` 2 → 3 ✓ ADOPTED

**Result:** zero regressions across all six paired comparisons — 0 of 144 row comparisons moved
backwards. substring 0.9014 → 0.9577, citation 0.9178 → 0.9452, for +7.8% cost and +2.2% latency.

**The pre-registered criterion did not fire, and that is the more useful finding.**
`gold_coverage_avg` — the metric the whole pass existed to make decisive — moved by *nothing*.
The reason is an instrument gap, not an absent effect: **page-level gold lists are narrower than
the set of pages that state each fact.** The arm fixed one row by reading a page that states the
answer but isn't in that row's gold, and coverage scored the win as zero. The pinned
`gold_coverage_avg` is therefore a lower bound.

**Auditing the four substring flips leaves two clean wins, not four** — one was a hedge
("either 108M or 110M", which passes a substring check while getting vaguer) and one was phrasing
luck. Adopted on the audited count, and recorded that way deliberately.

**Adaptive rerank ✗ rejected, narrowly.** Cheaper and the only arm to move coverage, but it gave
up a substring win and a citation fix, and its median latency was *worse* than fixed k=3. Kept
as a knob, not a default.

---

## Interpretability

### The heatmap overlay is real but weak — and nine of ten "fixes" made it worse ✓ ADOPTED (smoothing)

**Question:** the viewer's **"why this page?"** toggle overlays ColQwen2's query→patch MaxSim
grid. It looked like noise — on `sales_report.pdf` p1 the *bars* were the coldest region on the
page and the blank background was hot. Was the overlay wrong, or is the signal just weak?

**The instrument had to be built first, and it is free.** `answer_contains` from
`eval/dataset.jsonl` locates the answer's own line on the gold page via `pdftohtml -xml` — the
text layer the retriever deliberately never reads, exactly as `scripts/find_in_pdfs.py` already
uses it for labeling. That gives **44 answer regions over 19 pages** with no API key and no
judgement calls. The score is the patch grid's **ROC AUC for the answer region** (positive = a
patch overlapping it). It is deliberately threshold-free and **invariant to any monotone
renormalization**, so it measures the *reduction* and cannot be gamed by changing the colour ramp.

Two forward passes per page are cached as the full `(query_tokens, n_x, n_y)` tensor rather than a
reduced grid, so every candidate below was scored **offline for free** — the same trick
`probe_k50_retrieval.json` plays for slate policy.

**Result: the shipped reduction is not the best by raw AUC, but the better-scoring candidate was rejected.**

| reduction | mean AUC | AUC excl. blank patches |
|---|---|---|
| mean over content tokens | 0.6766 | 0.6789 |
| **`amax` over query tokens (shipped)** | **0.6620** | **0.6669** |
| `amax` over content tokens only | 0.6552 | 0.6597 |
| per-token z-score, then `amax` | 0.6115 | 0.6169 |
| minus the per-patch baseline | 0.5529 | 0.5571 |
| MaxSim decomposition (score of the tokens a patch wins) | 0.5222 | 0.5238 |
| per-patch z-score | 0.3746 | 0.3770 |
| *control:* ink density | 0.7642 | 0.6372 |

**The obvious diagnosis was wrong.** ColQwen2 appends 10 `<|endoftext|>` expansion tokens to every
query, and they win **44.9%** of all patches on the BLEU page — a textbook "the padding tokens are
matching the background" story. Dropping them changes the map *almost not at all* (0.662 → 0.655),
because the bias is per-**patch**, not per-token. Removing that baseline is worse still, and
per-patch z-scoring is far **below chance** — it inverts the signal. The most principled candidate,
decomposing the page's actual MaxSim score by which patch wins each query token, is barely above a
coin flip.

**The ink control is what makes the number interpretable.** A trivial "where is the ink" map scores
0.764 — *better* than the heatmap — because answer regions are always inky and margins never are.
But `corr(map, ink) = 0.001`: the heatmap is not an ink detector. Restrict to inky patches and ink
density collapses to 0.637 while the heatmap holds **0.667**. Among the patches that carry content,
the query similarity is the better predictor. The signal is genuine and query-specific.

**What did work was denoising, and only because it is not a renormalization.** A Gaussian blur over
the patch grid moves the patch *ranking*, so unlike a colour-ramp change the AUC can judge it:

| σ | mean AUC | items improved | sign-test p | *blurred random map* |
|---|---|---|---|---|
| 0 (raw) | 0.6620 | — | — | 0.477 |
| 1.0 | 0.7456 | 36/44 | 1.3e-05 | 0.469 |
| 1.25 | 0.7526 | 36/44 | 1.3e-05 | 0.475 |
| **1.5** | **0.7559** | **36/44** | **1.3e-05** | 0.477 |
| 1.75 | 0.7552 | 34/44 | 1.9e-04 | 0.476 |
| 2.0 | 0.7521 | 33/44 | 6.3e-04 | 0.473 |

The last column is the confound control, and it is the reason to believe the rest: blurring spreads
mass into contiguous regions, and the labels *are* contiguous, so the gain could have been
mechanical. Applying the identical blur to a **random** map leaves it at chance (0.477). The
optimum is a plateau over σ 1.25–1.75, not a knife edge. Adopted at `HEATMAP_SMOOTH_SIGMA=1.5`;
`0` restores the raw grid.

**What this does not fix.** AUC 0.756 is a real signal, not a strong one, and the overlay still
warms blank margins. The visual improvement is large — coherent regions instead of speckle — but
the honest reading of the feature is "these patches ranked highest for your query", not "the answer
is here". That is why the README caption says so. The crop, not the heatmap, is the claim about
where the answer was read.

**Also ruled out, cheaply:** the grid is not transposed (24×31 for a portrait page, aspect 0.774
against the page's 0.708), and **no display normalization helps** — rank, rank+gamma and percentile
clipping were all rendered and rejected by eye. They cannot help by construction: the noise is in
the ranking, and a monotone transform cannot reorder it. Percentile clipping actively makes it
worse by promoting isolated margin spikes into solid red.


## Ingest

### Visual-token budget ✗ REJECTED

**Question:** the embed step is ~98% of a page's ingest cost. Can it be cut?

**The reframe:** the previous pass measured *which stage* was slow but never measured *inside*
it. Splitting embed into sub-stages changed the conclusion — page-at-a-time is not what makes
ingest slow, **755 visual tokens per page** is. (It also revealed that the Qdrant upsert had
*never been measured at all*: profiling `--store` was opt-in, so an entire ingest-optimisation
pass ran with `upsert_measured: false`.)

ViT attention is quadratic in patch count, so the budget pays superlinearly:

| budget | patches | forward | speedup |
|---|---|---|---|
| **768 (checkpoint default)** | 755 | 6.48 s | 1.00× |
| 512 | 486 | 3.18 s | **2.04×** |
| 384 | 385 | 2.60 s | 2.49× |
| 256 | 263 | 1.64 s | 3.95× |

**The arm:** 512 tokens, against a control that re-scored the pinned index under the new code
and reproduced the baseline exactly.

| metric | 768 (control) | 512 | Δ |
|---|---|---|---|
| recall@1 | 0.6712 | 0.6438 | −0.0274 |
| **recall@12** | **1.0000** | **0.9315** | **−0.0685** |

**Rejected because the *shape* of the degradation is the finding.** 10 rows improved and 5 lost
gold from the slate entirely — and **4 of those 5 are `table` rows**. At 486 patches the model can
no longer resolve digits in a dense numeric table. Dense-table reading is precisely what pays for
the speed. 384 and 256 were **not run**: they strictly reduce information reaching the model, so
they cannot pass a bar that 512 already fails.

**So the knob ships, not the number.** A corpus of prose would likely take this trade happily;
this eval cannot see that case.

### Batching on Apple Silicon ✗ REJECTED (twice over)

**The bug, found by an equivalence gate rather than a test.** On MPS + bfloat16, a batched forward
pass **silently corrupts the first sequence in the batch**. Slot 0 comes back ~0.4 per component
from its solo embedding; slots 1..n−1 are bit-identical. CPU float32 and MPS float32 are exact, so
the batching code is fine and the bf16 kernel is not.

Nothing about it is visible from outside: no error, no NaN, correct patch counts, and an ingest that
records the document as current. It would poison one page in every batch, and **no test that stubs
the model could see it.**

**The corruption is fixable** — prepend a throwaway page, discard output index 0. Verified
bit-identical:

```text
naive batch of 3 vs solo:   slot 0 delta 0.411133  <-- CORRUPT
                            slot 1 delta 0.000000
sacrificial pad, slot 0 discarded:  all three pages delta 0.000000
```

**It was not implemented, because batching is slower than batch-1 on this hardware anyway**, before
paying anything for the wasted slot:

| batch | median s/useful page | vs batch 1 |
|---|---|---|
| **1** | **8.367** | **1.000×** |
| 2 | 11.539 | 0.725× |
| 4 | 10.921 | 0.766× |

So `embedder._batching_is_supported` refuses to batch on MPS at all — the blunt device check, which
turns out to cost this hardware nothing. Do not "fix" it without re-running that benchmark on the
target GPU.

### Three levers that look obvious and do nothing ✗

Recorded so they are not retried:

- **`RENDER_DPI` cannot speed up embedding.** A 150-DPI page is 2.1 MP, and `smart_resize` hands the
  model 672×868 — it already reads pages at ~79 DPI equivalent. DPI buys render time and disk only.
- **dtype is not it.** fp16 vs bf16 measured 6.5 s vs 6.5 s — inside this box's ~46% noise floor.
- **MRL does not apply, twice over.** ColQwen2-v1.0 has no matryoshka config, and it targets the
  1536→128 projection rather than the 2.25B backbone, so it cannot move the forward pass at all.
  As a storage lever, binary quantization already beats it 32× to 2–4×.

### Moving CPU work off the GPU's thread — worth less than it looks

Measured as **GPU-busy fraction, not s/page**, because s/page spreads 50% *within* a single arm here
while the ratio survives thermal drift.

| arm | GPU idle | rounds won |
|---|---|---|
| serial | 9.7% | — |
| store worker only | 10.0% | **0 of 3** |
| + preprocess lookahead | **7.0%** | **3 of 3** |

0.61 s/page of serial work bought only 2.7 points of GPU idle, because **background Python threads
contend with the GPU dispatch thread for the GIL**. The store worker is kept on probation — its
share grows as the forward pass shrinks — and flagged in its own docstring as the piece to delete
if it stays flat.

---

## The instrument

### Confidence calibration — the pass that overturned its own premise

The backlog carried this as a small UX cleanup: *"the confidence signals carry almost no
information (measured) — either calibrate them or stop showing them."* **That verdict had never
actually been measured.** The thing it condemned worked; the thing nobody suspected was broken.

**1. `confidence_separation` was reported to four decimals off one row.** The pinned baseline has
73 answerable rows: 70 correct citations, 2 declines that cite nothing, and **one** wrong citation.
Its `−0.0062` is that single row minus the mean of the other 70. Earlier baselines report the *same
quantity with the opposite sign* at the same tiny n. Three passes read a sign flip in noise as a
finding.

Structurally: **this metric's negative class is the pipeline's own mistakes**, so it degrades exactly
as the system it measures improves.

**2. `self_conf_low_acc = 0.0` is a tautology**, and had been read as calibration twice. The
answerer pins `confidence = "low"` onto every not-found answer, and a declined answerable row scores
wrong — so the bucket is forced to 0.0 by the code.

**3. The formula was fine; the presentation was the defect.** Retrieval confidence is a softmax share
over `RETRIEVE_K` candidates, so `1/12 = 8.3%` is the *uniform* reference, not zero. The UI rendered
it as a raw percentage, so a maximally decisive retrieval displayed **"21%"** — the product appearing
to doubt answers it had got right.

**The lever: score the same formula against a label that has a negative class.** Pairing it with
`gold_rank` instead of `citation_correct` asks the same question where the negative class is recall@1
misses — deterministic, no API key, **24 negatives against 1**.

| | |
|---|---|
| `decisiveness_separation` | **+0.0127** (n = 49 hit / 24 miss) |
| AUC | **0.629** (0.5 = coin flip) |
| permutation p, one-sided, 20 000 shuffles | **0.016** |

**The signal carries information — weakly, but not by chance.** That is the opposite of what the
backlog concluded from 1–3 rows, and it is why the fix was to the presentation rather than the
formula. A difference of means alone could not have said this; AUC and the permutation test are what
turn +0.0127 into a verdict. The UI now shows it in the trace as `1.42× uniform`, not as a headline
chip — trace placement is what AUC 0.629 supports.

**Shipped:** every calibration *comparison* is withheld below n=5 and ships with its counts.
`confidence_separation` is now `null` and will stay null until the eval has ≥5 wrong citations —
honest, not fixed.

### The label audit — a third of the cross-document slice was not cross-document

The open lead was *"`gold_coverage_avg` (0.825) trails its ceiling `candidate_coverage_avg` (0.850)
— rerank is losing a row it was offered."* It was **one row**, and the row was mislabelled:

```
$ uv run python scripts/find_in_pdfs.py "50K questions" --pdf donut
donut.pdf  p.8  tion4 and consists of 50K questions defined on more than 12K documents [44].
```

`donut.pdf` p8 answers *both* halves of that question, so the reranker spending three slots on one
document was correct, and the 0.5 was the label's fault. Auditing all 20 cross-document rows found
**6 defective** in two classes: three where one page states both halves, three where a half is
answerable from a document outside its gold.

The class that matters is invisible from the metrics. `xdoc-colbert-vs-colpali-dim` was **scoring
1.0** — a question one page can answer still scores perfectly whenever retrieval happens to reach
both documents, so `gold_doc_coverage` structurally cannot surface it. Only reading the corpus can.

**The obvious repair is backwards.** Adding the alternate source to a row's gold makes it a
three-document question that now needs all three, because coverage divides by the number of distinct
gold documents. Widening labels makes coverage *harder*. Both classes are the same defect — the
question does not require two documents — and the only fix is at the question.

All six were rewritten, each replacement fact verified single-document across all 19 PDFs. Result,
with **no change to `src/`**:

| | before | after |
|---|---|---|
| `gold_coverage_avg` | 0.825 | **0.925** |
| `candidate_coverage_avg` | 0.850 | **0.925** |
| rerank-side losses | 1 | **0** |
| `substring_accuracy` | 0.958 | 0.931 |
| `judge_accuracy` | 0.945 | 0.932 |

The control is the paired diff: over the 14 cross-document rows the relabelling did not touch,
`gold_doc_coverage` is 0.9286 → 0.9286, *0 improved, 0 regressed*. Two metrics went **down**,
because the replacement labels are stricter than the bare numbers they replaced. This is a corrected
instrument, not a better pipeline — and the lead it opened to chase was closed by disproving it.

Two tools came out of it: **`eval/rescore.py`**, which recomputes every label-derived metric from a
stored report so a relabelling costs no run at all (the docs had claimed this was possible since
`candidate_pages` was added; nothing implemented it), and **`scripts/audit_xdoc_labels.py`**, which
can decide 8 of the 20 rows automatically and is honest about the other 12 rather than guessing.

### The corpus is part of the instrument

The eval was once **saturated end-to-end**: on a 43-page corpus, `RETRIEVE_K=10` returned 23% of the
index per query, so recall was pinned at 1.0 and rerank recall, citation accuracy, substring match
and the judge all inherited the ceiling. The harness could not fail.

Growing the index to ~363 pages (16 distractor papers pinned by sha256, chosen to be *confusable*
with the gold documents) drops that to 2.8%. **Do not "simplify" the eval by removing the fetch
step** — without the distractors every downstream metric returns to 1.0 and the regression guard
stops guarding.

---

## Deployment

Everything above tunes the pipeline. These two are about the *deployment*, and both are defects
rather than arms — kept here because their shared property is the interesting part: **each one
fails silently**, which is why neither showed up in any metric above.

### The corpus was half-persisted (and the obvious repair was a no-op)

The `app` service mounted exactly one volume, the model cache. Page PNGs lived inside the
container; vectors lived in `qdrant_storage`. So any container recreate kept every vector and
destroyed every page — and what that produces is *not* an error. `GET /corpus` still listed all 19
documents (it reads Qdrant payloads), while every query dropped its whole slate for missing images
and answered "not found". One WARNING per dropped hit was the only trace.

The trap underneath it: `_sync` skipped a document whose content hash and embed version still
matched — which they did, since only the images were gone. **`python src/ingest.py` skipped exactly
the documents that most needed rebuilding.** A second trigger existed one layer down: `image_path`
was stored *absolute*, so relocating the directory broke the corpus identically.

Fixed as four changes, because the volumes alone leave it un-relocatable and un-repairable: volumes
for `page_images`/`pdfs`; `image_path` stored **relative** to `PAGE_IMAGES_DIR` (reads still accept
legacy absolute paths, so no re-ingest and **no `EMBED_VERSION` bump** — this is payload, not vector
data); `_sync` treating missing page images as stale, which makes a plain re-ingest the repair; and
`index_health()` reporting the split at boot and on `/health`.

**The guard matters more than the fix.** A change to the retrieval read path fails by *silently
dropping hits* — the same bug it fixes. Retrieval-only against the pinned baseline:

| Metric | Baseline | After |
|---|---|---|
| `recall@1` / `recall@3` / `recall@12` | 0.6712 / 0.8493 / 1.0 | identical |
| `candidate_coverage_avg` | 0.850 | 0.850 |
| `gold_rank`, **per row** | avg 2.0548 | avg 2.0548 — **73/73 unchanged, 0 flipped** |

Row-level, not summary-level. And since the pinned index was written with absolute paths, that run
doubles as the backward-compat proof.

### One upload froze every user — ~16× on the wait for the model ✅ ADOPTED

`/ingest` held the GPU lock across the whole build while `/query` contended for it. The fix is not
politer waiting; it is making the ingest **give the GPU back** at each page boundary — the one
moment in the loop it is idle.

**The first version of this measurement was a single run, and it did not reproduce** — 11.2× became
2.1× on a re-run, plus a failed ingest. That is the noise floor this repo already documented for
this box in the ingest-throughput pass, applied to a number published without meeting it. The
corrected framing reports the statistic the mechanism actually moves: end-to-end latency is
dominated by Gemini (15.7 s → 131.9 s of pipeline time for the *same* query within one session), so
a "speedup" from it mostly measures Gemini. What the gate changes is the **wait for the model**.
Three interleaved rounds, 13-page ingest, query fired 15 s in:

| round | queueing cost | ingest remaining (the pre-gate wait) |
|---|---|---|
| 1 | 3.7 s | 247.2 s |
| 2 | 8.4 s | 126.8 s |
| 3 | 8.2 s | 135.3 s |
| **median** | **8.2 s** | **135.3 s** — ≈**16×** |

The wait drops from *the rest of the document* to *about one page batch*, and it is stable across
rounds where the end-to-end figure is not.

**A stalled upsert used to kill the whole ingest.** The failing re-run died at page 8 of 20 on a
Qdrant upsert hitting the client's invisible 60 s default. `_StoreWorker`'s failure aborts the
document, so one slow HTTP call discarded every page embedded so far. The obvious hypothesis — the
gate parking the ingest long enough for the idle connection to be reaped — was **tested and
refuted** (70 s idle, next upsert 0.02 s). The fix stands on its own regardless: `QDRANT_TIMEOUT_S`
is stated rather than inherited, and `upsert_pages` retries transient transport failures the way
`gemini_client` already does, safely, because point ids make the write idempotent.

Two further behaviours moved into `src/gpu_arbiter.py` rather than staying per-endpoint: a bounded
wait (503 + `Retry-After` past `GPU_WAIT_TIMEOUT_S`), and **disconnect safety, which was a live
bug** — `await asyncio.to_thread(...)` cancels the awaiting task, never the thread, so a client
going away released the model out from under its own running forward pass. Both are regression
-tested by *breaking* them: stub the gate and the ordering test fails; remove the shield and the
disconnect test fails.

---

## The interface

### The shell only worked on a wide desktop ✓ FIXED

`grep -c "@media" ui/src/theme.css` was **0**, and `.app` was a hard
`grid-template-columns: 220px 1fr 1.15fr`. Below ~1000px the two content columns fell under
400px each; below ~800px the app was unusable. Three layouts now — 1100px moves the rail
into a drawer, 820px collapses to one column with a `Session | Page` switcher so the active
pane keeps the full height. Nothing is dropped at any width.

The breakpoints are CSS-only: no `matchMedia` listener exists, because the drawer's transform
and its scrim are both media-queried, so a drawer left open across a resize to a wide viewport
simply becomes the static column again. One source of truth per breakpoint.

Two decisions worth recording. The closed drawer needs `visibility: hidden` and not just
`translateX(-100%)` — a translated-away element is still in the tab order, so the first Tab
past the hamburger vanishes into a drawer nobody can see. And the drawer is a **disclosure,
not a modal**: it gets `aria-expanded`, Esc and focus restore, but no `role="dialog"` and no
Tab trap, because the element it controls is the *same* element that is a static landmark
above 1100px, and a role conditional on viewport width would need exactly the JS media query
the CSS-only design avoids. The missing trap is an accepted cost, not an oversight.

### The cited page was cropped, and the citation box with it ✓ FIXED

The defect the document-viewer guard was written for, alive in the component that guard did
not cover. `.page-frame` carried `max-height: 100%`, which reads as "fit" and is not: for a
content-sized box with `overflow: hidden` a max-height does not resize anything, it clips.
Measured on the running app at 1280×860 with a real answer on screen:

| | measured |
|---|---|
| `.stage` | 567 × 265 |
| `.page-frame` | 412 × **213** |
| its `<img>` | 412 × **533** — 320px thrown away |
| `.box-overlay` | 23px tall, where the model's own box was 16% of 533 |

The cropping is the visible half. The dangerous half is that the overlay is positioned in
percentages of that same frame, so the citation was drawn against a rectangle the model never
measured. The mechanism differs from the dialog's, which is why a wider version of the first
guard would not have found it: `.stage` is `flex: 1` in a column that also holds the crop strip
and the candidate rail, so **the more regions an answer found, the less height the page got**.
Four regions took it to a ~20px band.

Fixed with the pattern `.doc-page` already followed, reusing the same helper
(`lib.pageFrameStyle`): an inline aspect ratio from the image's natural size, which makes the
max constraints resize rather than clip. `.crops-strip` also stopped wrapping — with the
clipping fixed but the wrap left in, four regions still left the page 73px tall, complete and
unreadable.

**This is the second time the page frame has been the bug, and the second time a green suite
hid it.** Any change to page-frame geometry gets driven in a real browser before it is called
done.

### Accessibility stopped at the dialog ✓ FIXED

`DocumentModal` was exemplary and everything else had nothing: no landmarks in any file, no
`<h1>`, a toast with no live region, no focus ring on controls that had not defined their own,
and a real keyboard bug — confirming a delete unmounted the focused `✕` and dropped focus to
`<body>`, throwing the user to the top of the page mid-decision.

The one item here that came from measurement rather than a checklist: tabbing the running app
recorded **38 controls** between the first tab stop and the question box, because the rail
renders two per document across the 19-document corpus. Landmarks do not help there — they are
a screen-reader affordance and that is a sighted keyboard-only path — so there is a skip link.

Also fixed: two states that rendered as *nothing*. `corpus === null` was indistinguishable from
an empty corpus, and `refreshCorpus` handled only 401, so a 503 or a 429 left the rail blank
forever with no message and no retry.

## What's still open

- **The "why this page?" overlay is a weak signal, honestly labelled.** ROC AUC 0.756 for the
  answer region after smoothing (0.662 before), against 0.5 for chance. Nine alternative
  reductions all scored worse. It still warms blank margins, and nothing measured so far fixes
  that — the remaining ideas are a different granularity (the patch grid is only ~24×31) or a
  different checkpoint, both of which are re-embeds rather than knobs.
- **Nothing regression-guards that 0.756.** The probe needs the model, and CI has neither GPU nor
  checkpoint. Committing the 44 cached similarity tensors (~40 KB each) would turn the sweep into
  a real offline gate — the same move `probe_k50_retrieval.json` made for slate policy.
- ~~**`gold_coverage_avg` trails its ceiling — rerank is losing a row it was offered.**~~
  **Disproved.** It was one row, and that row was mislabelled: `donut.pdf` p8 states both halves of
  the question, so the reranker was right to spend three slots on one document. Auditing all 20
  cross-document rows found **6 defective**, one of which was scoring 1.0. See
  [the label-audit pass](ENGINEERING_LOG.md). After the rewrite, `gold_coverage_avg ==
  candidate_coverage_avg == 0.925` and rerank-side losses are **0**.
- **Three metrics are back on the ceiling** (`recall@12`, `rerank_recall`, `abstention_accuracy`)
  with no observed headroom in the current baseline — while their configured gates remain active
  and can fail after regression, three of the ten gates are currently un-trippable. Re-de-saturating
  needs new *question types*, not a bigger corpus.
- **One row needs a relevance-aware cap.** A rank-based cap spends `donut.pdf`'s quota on four
  non-gold pages. This is the honest failure mode of capping by rank.
- **`scripts/audit_xdoc_labels.py` can only decide 8 of the 20 cross-document rows.** The other 12
  carry reference labels too generic to locate a page — bare numbers (`"6"`, `"50"`) or words spread
  across the corpus (`"English"` is in 11 of 19 PDFs). They were audited by hand once and are not
  regression-guarded. Discriminating reference strings would turn the script into a real gate.
- **The splitter reaches ~40% of natural phrasings.** "For both X and Y…", "Compare X with Y" and
  semicolon-joined questions are not split. Reach, not accuracy, is the reason to consider an LLM
  splitter.
- **`_RRF_K = 60` is untuned** for slates this shallow — it is calibrated for TREC lists thousands
  deep, and flattens ranks 1–4 to within ~5% across a 12-page slate.
- **The shipped confidence statistic is the weakest one that works — swap scoped, unspent.** Tested
  offline against `calib_baseline.json` (73 rows, 24 negatives, label `gold_rank == 1`):
  `zscore_top1` **AUC 0.8078**, `margin_mean` 0.7908, `margin_rel`/`ratio_top2` 0.7874, the shipped
  `softmax_top1` **0.6293**, and score entropy **0.4949** — no signal at all. All p < 0.0001 except
  the shipped formula. The harness reproduces the documented 0.629 to four decimals, which is what
  validates it. Not adopted in the label-audit pass: it changes a user-facing number and the eval's
  `top1_decisiveness`, so it wants its own arm with a live confirmation.
- **A CUDA box would change three conclusions at once** — batching, the GIL contention, and the
  token-budget trade all measured against a saturated MPS device.
- **`EMBED_VISUAL_TOKENS` has never been swept on a mostly-prose corpus**, which is the case where
  it would be free speed.
- **`/images` is unauthenticated with guessable paths** (`<stem>_page_<n>.png`). Deliberate — an
  `<img src>` cannot send a header — and fine for a demo corpus, but a confidentiality leak the
  moment real users upload their own PDFs. Signed expiring URLs would close it without breaking
  `<img>`.
- **One shared `API_KEY` means one shared corpus**: every holder can `DELETE` every other holder's
  documents. Per-key corpus scoping is the middle ground short of real tenancy.
- **Nothing bounds total Gemini spend.** Per-IP rate limits do not — 50 addresses at 30/min each is
  unbounded. `request_context` already computes `est_cost_usd` per call, so a daily ceiling with a
  kill switch is cheap at the same choke point.
- **`GPU_WAIT_TIMEOUT_S = 60` is a judgement, not a derivation.** It wants to sit above one gated
  page batch and below a client's own timeout; neither bound was measured.
- ~~**CI never builds the Docker image.**~~ **Closed** — a `docker` job now validates
  `docker-compose.yml` and builds the image on every PR.
