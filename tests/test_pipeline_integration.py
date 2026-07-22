"""Full-graph integration tests: retrieve -> rerank -> answer -> highlight.

These exercise the *compiled* LangGraph flow (`build_graph().invoke`) end to end,
with every external boundary stubbed at the name-as-imported choke points -
`graph.embed_query` / `graph.search`, `reranker.generate` / `reranker._candidate_part`,
`answerer.generate` / `answerer.image_part`, and `graph.crop_region` /
`graph.annotate_page`. No model, API key, network, or PNGs - the suite verifies
*wiring* (state flow, index alignment, degradation paths), not pixels.

`graph.rerank` is re-bound to the real `reranker.rerank` with `k=2` pinned, so the
tests don't depend on the RERANK_K value a local .env happens to set.
"""

from types import SimpleNamespace

import pytest

from src import answerer, graph, reranker
from src.answerer import Citation, Region


def _cite(answer: str, found: bool, source_page: int, box: list[int]) -> Citation:
    """A single-region Citation, mirroring the old (source_page, box) shorthand."""
    regions = [Region(source_page=source_page, box=box)] if found else []
    return Citation(answer=answer, found=found, regions=regions)


def _hit(n: int) -> dict:
    """A minimal Qdrant hit for page n of the fake corpus."""
    return {
        "pdf": "doc.pdf",
        "page_number": n,
        "image_path": f"p{n}.png",
        "score": round(1.0 - n / 100, 4),
    }


CANDIDATES = [_hit(n) for n in range(1, 11)]


def _rerank_response(indices: list) -> SimpleNamespace:
    return SimpleNamespace(parsed=SimpleNamespace(page_indices=indices), text="")


def _answer_response(citation: Citation) -> SimpleNamespace:
    return SimpleNamespace(parsed=citation, text="")


@pytest.fixture
def pipeline(monkeypatch):
    """A compiled graph with all boundaries stubbed; `calls` records highlight IO."""
    calls = {"cropped": [], "annotated": []}
    monkeypatch.setattr(graph, "embed_query", lambda q: [[0.0] * 128])
    monkeypatch.setattr(graph, "search", lambda vec: [dict(h) for h in CANDIDATES])
    monkeypatch.setattr(graph, "rerank", lambda q, pages: reranker.rerank(q, pages, k=2))
    monkeypatch.setattr(reranker, "_candidate_part", lambda path: f"thumb:{path}")
    monkeypatch.setattr(answerer, "image_part", lambda path: f"img:{path}")

    def crop(path, box, index=0):
        calls["cropped"].append((str(path), list(box)))
        return "crop.png"

    def annotate(path, boxes):
        calls["annotated"].append((str(path), [list(b) for b in boxes]))
        return "annotated.png"

    monkeypatch.setattr(graph, "crop_region", crop)
    monkeypatch.setattr(graph, "annotate_page", annotate)
    compiled = graph.build_graph()
    return SimpleNamespace(invoke=lambda q="q": compiled.invoke({"question": q}), calls=calls)


def test_happy_path_flows_rerank_order_into_answer_and_highlight(pipeline, monkeypatch):
    monkeypatch.setattr(reranker, "generate", lambda **kw: _rerank_response([3, 1]))
    cite = _cite("42", True, 1, [100, 100, 200, 200])
    monkeypatch.setattr(answerer, "generate", lambda **kw: _answer_response(cite))

    state = pipeline.invoke()

    assert [h["page_number"] for h in state["retrieved"]] == [3, 1]
    assert state["answer"] == "42"
    assert state["citation"]["source_page"] == 1
    assert state["crop_path"] == "crop.png"
    assert state["annotated_path"] == "annotated.png"
    # The load-bearing invariant: source_page=1 indexes the RERANKED list, so the
    # highlight must be cut from page 3's image (rerank winner), not page 1's.
    assert pipeline.calls["cropped"] == [("p3.png", [100, 100, 200, 200])]
    assert pipeline.calls["annotated"] == [("p3.png", [[100, 100, 200, 200]])]


def test_candidates_holds_full_retrieval_and_survives_rerank(pipeline, monkeypatch):
    monkeypatch.setattr(reranker, "generate", lambda **kw: _rerank_response([3, 1]))
    cite = _cite("42", True, 1, [100, 100, 200, 200])
    monkeypatch.setattr(answerer, "generate", lambda **kw: _answer_response(cite))

    state = pipeline.invoke()

    # Pre-rerank top-k is preserved for eval/UI even though rerank overwrites
    # `retrieved` down to 2 pages.
    assert [h["page_number"] for h in state["candidates"]] == list(range(1, 11))
    assert state["candidates"] == CANDIDATES
    assert len(state["retrieved"]) == 2


def test_rerank_failure_falls_back_to_qdrant_top_k(pipeline, monkeypatch):
    def boom(**kw):
        raise RuntimeError("quota")

    monkeypatch.setattr(reranker, "generate", boom)
    cite = _cite("ok", True, 1, [0, 0, 10, 10])
    monkeypatch.setattr(answerer, "generate", lambda **kw: _answer_response(cite))

    state = pipeline.invoke()

    # Degraded rerank -> Qdrant score order, pipeline still completes end to end.
    assert [h["page_number"] for h in state["retrieved"]] == [1, 2]
    assert state["crop_path"] == "crop.png"
    assert pipeline.calls["cropped"] == [("p1.png", [0, 0, 10, 10])]


def test_rerank_garbage_indices_are_cleaned_before_answer(pipeline, monkeypatch):
    # 99 / 0 out of range, 3 duplicated -> _valid_order keeps 3, tops up with 1.
    monkeypatch.setattr(reranker, "generate", lambda **kw: _rerank_response([99, 0, 3, 3]))
    cite = _cite("ok", True, 2, [0, 0, 10, 10])
    monkeypatch.setattr(answerer, "generate", lambda **kw: _answer_response(cite))

    state = pipeline.invoke()

    assert [h["page_number"] for h in state["retrieved"]] == [3, 1]
    # source_page=2 -> second reranked page (page 1).
    assert pipeline.calls["cropped"] == [("p1.png", [0, 0, 10, 10])]


def test_answer_failure_degrades_to_not_found_and_skips_highlight(pipeline, monkeypatch):
    monkeypatch.setattr(reranker, "generate", lambda **kw: _rerank_response([3, 1]))

    def boom(**kw):
        raise RuntimeError("api down")

    monkeypatch.setattr(answerer, "generate", boom)

    state = pipeline.invoke()

    assert state["citation"]["found"] is False
    assert state["citation"]["source_page"] == 0
    assert state["crop_path"] is None
    assert state["annotated_path"] is None
    assert pipeline.calls["cropped"] == []


def test_malformed_answer_json_degrades_to_not_found(pipeline, monkeypatch):
    monkeypatch.setattr(reranker, "generate", lambda **kw: _rerank_response([3, 1]))
    monkeypatch.setattr(
        answerer, "generate",
        lambda **kw: SimpleNamespace(parsed=None, text="not json at all"),
    )

    state = pipeline.invoke()

    assert state["citation"]["found"] is False
    assert state["crop_path"] is None


def test_empty_retrieval_completes_cleanly(pipeline, monkeypatch):
    monkeypatch.setattr(graph, "search", lambda vec: [])
    not_found = _cite("No pages indexed.", False, 0, [])
    monkeypatch.setattr(answerer, "generate", lambda **kw: _answer_response(not_found))
    # rerank must pass [] through without a Gemini call; make one loud if attempted.
    def no_call(**kw):
        raise AssertionError("rerank should not call Gemini on empty retrieval")

    monkeypatch.setattr(reranker, "generate", no_call)

    state = pipeline.invoke()

    assert state["retrieved"] == []
    assert state["candidates"] == []
    assert state["crop_path"] is None
