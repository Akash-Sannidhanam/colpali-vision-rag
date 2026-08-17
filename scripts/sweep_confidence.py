"""Offline sweep of alternative retrieval-confidence statistics.

Every candidate below is a pure function of the slate's MaxSim score distribution, and
`eval/reports/calib_baseline.json` stores per-candidate `score` alongside `gold_rank`. So
each can be scored against the same label the shipped formula is measured on, with **no
live arm, no GPU and no API key** - which is the point: a confidence formula is one of the
few things in this pipeline that can be evaluated entirely from a stored report.

Anchored on `gold_rank == 1` rather than `citation_correct` on purpose. The citation-based
pairing's negative class is the pipeline's own wrong citations - one row on the pinned
baseline, which is why `confidence_separation` is withheld. This one has 24.

Result on the shipped corpus (see docs/EXPERIMENTS.md): `zscore_top1` AUC **0.8078** against
the shipped `softmax_top1`'s **0.6293**, with score entropy at 0.4949 - no signal at all.
The shipped formula reproduces its documented 0.629 here, which is what validates the
harness. The swap is scoped but unspent: it changes a user-facing number.

    uv run python scripts/sweep_confidence.py
    uv run python scripts/sweep_confidence.py --report eval/reports/other.json

Needs a report written after the confidence-calibration pass; earlier ones carry no
per-candidate `score` and the sweep exits 1 rather than scoring zeros.
"""
import argparse
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path

DEFAULT_REPORT = Path(__file__).resolve().parent.parent / "eval/reports/calib_baseline.json"


# --- candidate statistics: slate scores (descending) -> a scalar "decisiveness" ---

def softmax_top1(s):
    """The shipped formula: mean-scaled softmax mass on the top page."""
    n = len(s)
    mean = sum(s) / n
    if mean <= 0:
        return 1.0 / n
    scaled = [x / mean for x in s]
    hi = max(scaled)
    e = [math.exp(x - hi) for x in scaled]
    return e[0] / sum(e)


def margin_rel(s):
    """Relative top1-vs-top2 gap. The obvious alternative, and the cheapest."""
    return 0.0 if len(s) < 2 or s[0] <= 0 else (s[0] - s[1]) / s[0]


def margin_mean(s):
    """Top1-vs-top2 gap in units of the slate mean - scale-invariant like the softmax."""
    n = len(s)
    mean = sum(s) / n
    return 0.0 if len(s) < 2 or mean <= 0 else (s[0] - s[1]) / mean


def neg_entropy(s):
    """Negated Shannon entropy of the mean-scaled softmax: peaked slate -> high value."""
    n = len(s)
    mean = sum(s) / n
    if mean <= 0:
        return 0.0
    scaled = [x / mean for x in s]
    hi = max(scaled)
    e = [math.exp(x - hi) for x in scaled]
    tot = sum(e)
    p = [x / tot for x in e]
    return -(-sum(x * math.log(x) for x in p if x > 0)) / math.log(n)


def zscore_top1(s):
    """How many slate standard deviations the top page sits above the mean."""
    if len(s) < 2:
        return 0.0
    sd = statistics.pstdev(s)
    return 0.0 if sd == 0 else (s[0] - statistics.fmean(s)) / sd


def ratio_top2(s):
    return 0.0 if len(s) < 2 or s[1] <= 0 else s[0] / s[1]


STATS = {
    "softmax_top1 (shipped)": softmax_top1,
    "margin_rel": margin_rel,
    "margin_mean": margin_mean,
    "neg_entropy": neg_entropy,
    "zscore_top1": zscore_top1,
    "ratio_top2": ratio_top2,
}


def auc(pos, neg):
    """Mann-Whitney AUC: P(a random positive scores above a random negative)."""
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a, b in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def perm_p(pos, neg, trials=20000, seed=0):
    """Two-sided permutation p-value on the AUC, so no distributional assumption."""
    rng = random.Random(seed)
    observed = abs(auc(pos, neg) - 0.5)
    pool = pos + neg
    hits = 0
    for _ in range(trials):
        rng.shuffle(pool)
        a = auc(pool[: len(pos)], pool[len(pos):])
        hits += abs(a - 0.5) >= observed
    return (hits + 1) / (trials + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--trials", type=int, default=20000,
                        help="permutation shuffles per statistic (default 20000)")
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text())
    rows = [
        r for r in report["rows"]
        if r.get("candidate_pages") and "gold_rank" in r
        and all(c.get("score") is not None for c in r["candidate_pages"])
    ]
    if not rows:
        print(f"{args.report} carries no per-candidate scores - nothing to sweep. Reports "
              f"written before the confidence-calibration pass do not store them.",
              file=sys.stderr)
        return 1

    print(f"{len(rows)} scored rows from {args.report}")
    hit = [r for r in rows if r["gold_rank"] == 1]
    print(f"positives (gold_rank==1): {len(hit)}   negatives: {len(rows) - len(hit)}\n")
    print(f"{'statistic':24s} {'AUC':>7s} {'perm p':>8s}  {'mean(hit)':>10s} {'mean(miss)':>11s}")
    print("-" * 66)
    for name, fn in STATS.items():
        pos, neg = [], []
        for r in rows:
            s = sorted((float(c["score"]) for c in r["candidate_pages"]), reverse=True)
            (pos if r["gold_rank"] == 1 else neg).append(fn(s))
        a = auc(pos, neg)
        print(f"{name:24s} {a:7.4f} {perm_p(pos, neg, args.trials):8.4f}  "
              f"{statistics.fmean(pos):10.4f} {statistics.fmean(neg):11.4f}")
    print("\nAUCs are directly comparable; the p-values are two-sided permutation tests on\n"
          "|AUC-0.5| and are NOT the same construction as the 0.016 quoted in older write-ups.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
