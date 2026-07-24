"""Tests for the deterministic retrieval-decisiveness confidence (src.confidence).

Pure arithmetic over candidate MaxSim scores - no models, API key, or network.
"""

import math

from src.confidence import retrieval_confidence


def _cand(pdf: str, page: int, score: float) -> dict:
    """One retrieval candidate: source page plus its MaxSim score."""
    return {"pdf": pdf, "page_number": page, "score": score}


def test_none_when_no_candidates():
    """No candidates means there is nothing to be confident about."""
    assert retrieval_confidence([], {"pdf": "d.pdf", "page_number": 1}) is None


def test_none_when_cited_is_none():
    """A missing citation yields None rather than a fabricated score."""
    assert retrieval_confidence([_cand("d.pdf", 1, 5.0)], None) is None


def test_none_when_cited_not_in_candidates():
    """A cited page outside the candidate set can't be scored against it."""
    cands = [_cand("d.pdf", 1, 5.0), _cand("d.pdf", 2, 4.0)]
    assert retrieval_confidence(cands, {"pdf": "d.pdf", "page_number": 9}) is None


def test_single_candidate_is_full_confidence():
    """One candidate takes the whole softmax mass."""
    cands = [_cand("d.pdf", 1, 3.2)]
    assert retrieval_confidence(cands, cands[0]) == 1.0


def test_equal_scores_split_evenly():
    """Identical scores share confidence equally - no arbitrary winner."""
    cands = [_cand("d.pdf", i, 5.0) for i in range(1, 5)]  # four equal candidates
    assert math.isclose(retrieval_confidence(cands, cands[0]), 0.25)


def test_decisive_top_hit_dominates():
    """A clear score gap reads as high confidence."""
    cands = [_cand("d.pdf", 1, 20.0), _cand("d.pdf", 2, 5.0), _cand("d.pdf", 3, 4.0)]
    conf = retrieval_confidence(cands, cands[0])
    assert conf > 0.65  # clearly the winner


def test_flat_distribution_is_less_confident_than_decisive():
    """A barely-separated candidate set must not read as confident as a decisive one."""
    decisive = [_cand("d.pdf", 1, 20.0), _cand("d.pdf", 2, 5.0), _cand("d.pdf", 3, 4.0)]
    flat = [_cand("d.pdf", 1, 6.0), _cand("d.pdf", 2, 5.5), _cand("d.pdf", 3, 5.4)]
    # A barely-separated set must NOT read as confident as a decisive one.
    assert retrieval_confidence(decisive, decisive[0]) > retrieval_confidence(flat, flat[0])


def test_scale_invariant_across_query_length():
    """Confidence depends on score ratios, not magnitudes, so query length doesn't skew it."""
    # MaxSim magnitude grows with query token count; scaling every score by a
    # constant must leave the confidence unchanged.
    base = [_cand("d.pdf", 1, 10.0), _cand("d.pdf", 2, 6.0), _cand("d.pdf", 3, 4.0)]
    scaled = [_cand("d.pdf", 1, 100.0), _cand("d.pdf", 2, 60.0), _cand("d.pdf", 3, 40.0)]
    assert math.isclose(
        retrieval_confidence(base, base[0]),
        retrieval_confidence(scaled, scaled[0]),
        rel_tol=1e-9,
    )


def test_non_top_page_gets_less_than_top():
    """Citing a lower-ranked page yields lower confidence than citing the top one."""
    cands = [_cand("d.pdf", 1, 20.0), _cand("d.pdf", 2, 5.0), _cand("d.pdf", 3, 4.0)]
    assert retrieval_confidence(cands, cands[1]) < retrieval_confidence(cands, cands[0])
