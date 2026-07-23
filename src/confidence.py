"""Deterministic retrieval-decisiveness confidence from ColQwen2 MaxSim scores.

The retriever returns a per-page MaxSim score whose absolute magnitude grows with
the query's token count, so scores can't be read as probabilities directly. We
scale the candidate scores by their mean (scale-invariant across queries, while
preserving the top-vs-rest *margin* - a z-score would flatten that and call a
barely-separated set "confident"), take a softmax, and report the cited page's
share of the mass: how decisively retrieval preferred that page over the rest of
the candidate set.

This is a signal about *retrieval*, not answer correctness - surface it labelled as
such (e.g. "retrieval 78%"), alongside, never in place of, the model's own
self-reported answer confidence.
"""

import math


def retrieval_confidence(candidates: list[dict], cited: dict | None) -> float | None:
    """Fraction of the softmax mass on `cited` across the candidate MaxSim scores.

    `cited` is matched to a candidate by (pdf, page_number). Returns None when there
    are no candidates or the cited page isn't among them. All-equal scores (or a
    single candidate) yield 1/n - maximally undecided.
    """
    if not candidates or cited is None:
        return None

    key = (cited.get("pdf"), cited.get("page_number"))
    idx = next(
        (i for i, c in enumerate(candidates) if (c.get("pdf"), c.get("page_number")) == key),
        None,
    )
    if idx is None:
        return None

    scores = [float(c.get("score", 0.0)) for c in candidates]
    n = len(scores)
    mean = sum(scores) / n
    if mean <= 0:  # non-positive scores can't be mean-scaled; call it undecided
        return 1.0 / n

    scaled = [s / mean for s in scores]
    hi = max(scaled)                                   # shift for numerical stability
    exps = [math.exp(s - hi) for s in scaled]
    return exps[idx] / sum(exps)
