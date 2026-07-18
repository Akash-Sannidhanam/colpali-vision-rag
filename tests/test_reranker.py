"""Tests for the rerank step's pure ranking logic (src.reranker).

These exercise `_valid_order` and `rerank`'s early pass-through, so they need no
models, API key, or network - the Gemini call is never reached. One test stubs
the shared `gemini_client.generate` to assert the call is routed with RERANK_MODEL.
"""

from types import SimpleNamespace

from src import reranker
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


def test_rerank_routes_through_shared_client_with_rerank_model(monkeypatch):
    # With more pages than k, the Gemini call is reached: assert it goes through
    # gemini_client.generate tagged as a rerank call using RERANK_MODEL, and that
    # the model's page order is honored.
    calls: list = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(parsed=SimpleNamespace(page_indices=[2, 1]), text="")

    monkeypatch.setattr(reranker, "_candidate_part", lambda p: None)  # no PNGs read
    monkeypatch.setattr(reranker, "generate", fake_generate)

    pages = [_page(1), _page(2), _page(3)]
    out = rerank("q", pages, k=2)

    assert out == [pages[1], pages[0]]                       # model order [2, 1]
    assert calls and calls[0]["model"] == reranker.RERANK_MODEL
    assert calls[0]["purpose"] == "rerank"
    assert calls[0]["response_schema"] is reranker.Rerank
