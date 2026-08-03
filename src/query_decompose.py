"""Split a two-part question into sub-queries, and fuse their rankings.

Why this exists: ColQwen2's MaxSim on a two-part question is dominated by whichever
document matches more of the query tokens, and it takes the slate. `_diversify` caps
how many slots one document may hold, but a cap cannot conjure a page retrieval never
ranked at all - and on `xdoc-splade-vocab-dpr-dense` the second gold document sits
outside even the top-50. The only remaining lever is to stop asking one query.

Both functions are pure (strings and plain dicts in, strings and plain dicts out) so
they unit-test with no Qdrant, following the `vector_store._diversify` precedent.

**The splitter is deliberately conservative.** 63 of the 83 eval questions are
single-document, and on those a split is pure cost: it spends fetch depth on a half
that names nothing. Every rule below is a reason *not* to split.
"""

import re

from src.config import MAX_SUBQUERIES

# Clause boundaries, not conjunctions. Requiring a comma or a sentence break before
# "and" is what separates clause coordination ("<A>, and <B>?") from noun-phrase
# coordination ("document retrieval and question answering") - a bare " and " rule
# splits the latter and embeds "question answering" alone, which names no document in
# the corpus. The `(?<=\?)` lookbehind keeps the question mark on the left half.
_SPLIT_RE = re.compile(
    r",\s+and\s+"
    r"|,\s+but\s+"
    r"|(?<=\?)\s+"
    r"|\s+versus\s+"
    r"|\s+vs\.?\s+",
    re.IGNORECASE,
)

# A half shorter than this is not a searchable query - "why?" carries no document
# signal at all, and embedding it just adds noise to the fused pool.
_MIN_PART_TOKENS = 3

# Reciprocal-rank-fusion damping. 60 is the standard default from the original TREC
# work; left a module constant rather than a knob because nothing here wants to tune
# it, and the eval has no way to tell a good value from a bad one yet.
_RRF_K = 60


def decompose(question: str) -> list[str]:
    """The queries to embed for `question`: `[question]`, or `[question, A, B]`.

    The original question is **always** first and always present. It is not merely a
    fallback: a page that answers *both* halves is ranked highest by the whole
    question, and fusing only the halves would demote it. Keeping it makes
    decomposition strictly additive to today's ranking rather than a replacement.

    Returns at most `MAX_SUBQUERIES` parts beyond the original. Dropping a trailing
    clause is safe rather than arbitrary - since the whole question is in the fusion
    set, an uncapped clause is no worse off than it is today.
    """
    parts = [p.strip() for p in _SPLIT_RE.split(question)]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return [question]

    parts = parts[:MAX_SUBQUERIES]
    # One unusable half discards the whole split: half a question's candidates fused
    # against a noise ranking is worse than the undecomposed query.
    if any(len(p.split()) < _MIN_PART_TOKENS for p in parts):
        return [question]
    return [question, *parts]


def fuse_rrf(
    rankings: list[list[dict]], weights: list[float] | None = None
) -> list[dict]:
    """Merge per-sub-query rankings by reciprocal rank fusion, deduped on (pdf, page).

    `weights` scales each ranking's vote (default: all equal). It exists because equal
    voting has a specific failure here: RRF rewards agreement, so a page found by two
    rankings outranks one found well by a single ranking. Fusing the whole question
    alongside its halves therefore lets the query that *could not* find the second
    document out-vote the half that could - measured, it pushed `dpr.pdf` p3 out of the
    slate in favour of two `dpr.pdf` pages the whole question already liked.

    Rank-based fusion is not a stylistic choice. MaxSim *sums* max-similarity over the
    query's tokens, so a longer sub-query's scores are systematically larger than a
    shorter one's (see src/confidence.py, which documents the same property). Merging
    by raw score would hand the slate to whichever half happened to be wordier -
    recreating the monopolisation this module exists to break. RRF only reads position.

    A page found by several sub-queries accumulates their contributions, so agreement
    across halves is what earns a top slot.

    `score` is **overwritten with the fused score** whenever more than one ranking is
    merged: a slate carrying raw MaxSim from two different query scales side by side
    would corrupt `confidence.retrieval_confidence`, which mean-scales it. With a
    single ranking there is no scale conflict, so the raw scores pass through
    untouched - which, with the order also preserved, is what keeps the undecomposed
    path byte-identical to what it was before this module existed.
    """
    if not rankings:
        return []
    if len(rankings) == 1:
        return list(rankings[0])

    if weights is None:
        weights = [1.0] * len(rankings)
    fused: dict[tuple, float] = {}
    first_seen: dict[tuple, dict] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.get("pdf"), hit.get("page_number"))
            fused[key] = fused.get(key, 0.0) + weight / (_RRF_K + rank)
            # Keep the earliest sighting, so a page's payload comes from the highest-
            # priority ranking that found it (the whole question is always first).
            first_seen.setdefault(key, hit)

    # sorted() is stable and dicts preserve insertion order, so ties resolve to
    # first-seen order and the whole function stays deterministic.
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [{**first_seen[key], "score": round(score, 6)} for key, score in ordered]
