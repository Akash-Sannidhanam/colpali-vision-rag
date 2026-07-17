"""Tests for the rerank step's pure ranking logic (src.reranker).

These exercise `_valid_order` and `rerank`'s early pass-through, so they need no
models, API key, or network - the Gemini call is never reached.
"""

from src.reranker import _valid_order, rerank


def _page(n: int) -> dict:
    """A minimal retrieved-page dict; only identity matters for these tests."""
    return {"pdf": "doc.pdf", "page_number": n, "image_path": f"p{n}.png", "score": 1.0}


def test_model_order_is_preserved():
    # Gemini's chosen order wins over Qdrant score order.
    assert _valid_order([2, 1], 10, 2) == [2, 1]


def test_out_of_range_indices_are_dropped_then_topped_up():
    # 99/0/-1 are out of [1, 10]; only 2 survives, then top-up adds Qdrant #1.
    assert _valid_order([99, 0, -1, 2], 10, 2) == [2, 1]


def test_duplicates_are_removed():
    assert _valid_order([5, 5, 3], 10, 2) == [5, 3]


def test_bools_are_rejected_as_non_ints():
    # True/False are ints in Python; they must not count as page indices.
    assert _valid_order([True, 2], 10, 2) == [2, 1]


def test_fewer_than_k_tops_up_from_qdrant_order():
    assert _valid_order([3], 10, 2) == [3, 1]


def test_empty_raw_falls_back_to_qdrant_top_k():
    assert _valid_order([], 10, 2) == [1, 2]


def test_result_never_exceeds_available_pages():
    # Fewer pages than k -> returns exactly the pages that exist.
    assert _valid_order([], 1, 2) == [1]


def test_rerank_passthrough_when_k_ge_pages():
    # k >= number retrieved: no Gemini call, pages returned unchanged.
    pages = [_page(1), _page(2)]
    assert rerank("q", pages, k=2) == pages


def test_rerank_passthrough_on_empty_retrieval():
    assert rerank("q", [], k=2) == []
