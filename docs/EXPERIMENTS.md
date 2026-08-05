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

### The corpus is part of the instrument

The eval was once **saturated end-to-end**: on a 43-page corpus, `RETRIEVE_K=10` returned 23% of the
index per query, so recall was pinned at 1.0 and rerank recall, citation accuracy, substring match
and the judge all inherited the ceiling. The harness could not fail.

Growing the index to ~363 pages (16 distractor papers pinned by sha256, chosen to be *confusable*
with the gold documents) drops that to 2.8%. **Do not "simplify" the eval by removing the fetch
step** — without the distractors every downstream metric returns to 1.0 and the regression guard
stops guarding.

---

## What's still open

- **`gold_coverage_avg` (0.825) trails its ceiling `candidate_coverage_avg` (0.850)** — rerank is
  losing a row it was offered, for the first time since the slate pass closed that gap. The most
  concrete open lead.
- **Three metrics are back on the ceiling** (`recall@12`, `rerank_recall`, `abstention_accuracy`)
  with no observed headroom in the current baseline — while their configured gates remain active
  and can fail after regression, three of the ten gates are currently un-trippable. Re-de-saturating
  needs new *question types*, not a bigger corpus.
- **One row needs a relevance-aware cap.** A rank-based cap spends `donut.pdf`'s quota on four
  non-gold pages. This is the honest failure mode of capping by rank.
- **The splitter reaches ~40% of natural phrasings.** "For both X and Y…", "Compare X with Y" and
  semicolon-joined questions are not split. Reach, not accuracy, is the reason to consider an LLM
  splitter.
- **`_RRF_K = 60` is untuned** for slates this shallow — it is calibrated for TREC lists thousands
  deep, and flattens ranks 1–4 to within ~5% across a 12-page slate.
- **Alternative confidence statistics are now free to test** (top1-vs-top2 margin, score entropy)
  against the stored scores in `calib_baseline.json`, with no live arm required.
- **A CUDA box would change three conclusions at once** — batching, the GIL contention, and the
  token-budget trade all measured against a saturated MPS device.
- **`EMBED_VISUAL_TOKENS` has never been swept on a mostly-prose corpus**, which is the case where
  it would be free speed.
