"""Tests for src.answerer's graceful-degradation contract.

The answer step must never raise into the graph: a transient API error that
outlives the retries, a non-JSON response, or a valid-JSON-but-wrong-shape
response all degrade to a well-formed not-found citation. These stub the shared
`gemini_client.generate` (and `image_part`, so no PNGs are read), so they need no
model, API key, or network. The last test also covers the answer_node ->
highlight_node wiring that a bad response used to crash.
"""

from types import SimpleNamespace

from src import answerer


def _page(n: int) -> dict:
    """A minimal retrieved-page dict; image_part is stubbed so the path is unused."""
    return {"pdf": "doc.pdf", "page_number": n, "image_path": f"p{n}.png", "score": 1.0}


def _fake_generate(record, *, result=None, error=None):
    """A stand-in for gemini_client.generate that records kwargs and returns/raises."""
    def _gen(**kwargs):
        record.append(kwargs)
        if error is not None:
            raise error
        return result
    return _gen


def test_valid_citation_passes_through_and_routes_as_answer(monkeypatch):
    monkeypatch.setattr(answerer, "image_part", lambda p: None)
    calls: list = []
    citation = answerer.Citation(
        answer="42",
        found=True,
        regions=[answerer.Region(source_page=1, box=[10, 20, 30, 40])],
        confidence="high",
    )
    resp = SimpleNamespace(parsed=citation, text="")
    monkeypatch.setattr(answerer, "generate", _fake_generate(calls, result=resp))

    out = answerer.answer("q", [_page(1)])

    # regions is authoritative; source_page/box are derived from the first region.
    assert out == {
        "answer": "42",
        "found": True,
        "regions": [{"source_page": 1, "box": [10, 20, 30, 40]}],
        "source_page": 1,
        "box": [10, 20, 30, 40],
        "confidence": "high",
    }
    # routed through the shared client, tagged as an answer call
    assert calls and calls[0]["model"] == answerer.GEMINI_MODEL
    assert calls[0]["purpose"] == "answer"
    assert calls[0]["response_schema"] is answerer.Citation


def test_valid_text_json_is_parsed_when_sdk_parse_is_none(monkeypatch):
    monkeypatch.setattr(answerer, "image_part", lambda p: None)
    resp = SimpleNamespace(
        parsed=None,
        text='{"answer": "x", "found": true, "regions": [{"source_page": 2, "box": [1, 2, 3, 4]}]}',
    )
    monkeypatch.setattr(answerer, "generate", _fake_generate([], result=resp))

    out = answerer.answer("q", [_page(1), _page(2)])

    # confidence is omitted from the JSON, so it takes the schema's neutral default.
    assert out == {
        "answer": "x",
        "found": True,
        "regions": [{"source_page": 2, "box": [1, 2, 3, 4]}],
        "source_page": 2,
        "box": [1, 2, 3, 4],
        "confidence": "medium",
    }


def test_malformed_text_falls_back_to_not_found(monkeypatch):
    monkeypatch.setattr(answerer, "image_part", lambda p: None)
    resp = SimpleNamespace(parsed=None, text="definitely not json {{{")
    monkeypatch.setattr(answerer, "generate", _fake_generate([], result=resp))

    out = answerer.answer("q", [_page(1)])

    assert out == answerer._NOT_FOUND
    assert out is not answerer._NOT_FOUND   # a copy, not the shared constant
    assert out["found"] is False


def test_wrong_shape_json_falls_back_to_not_found(monkeypatch):
    monkeypatch.setattr(answerer, "image_part", lambda p: None)
    resp = SimpleNamespace(parsed=None, text='{"unexpected": "shape"}')
    monkeypatch.setattr(answerer, "generate", _fake_generate([], result=resp))

    assert answerer.answer("q", [_page(1)]) == answerer._NOT_FOUND


def test_generate_exception_falls_back_to_not_found(monkeypatch):
    monkeypatch.setattr(answerer, "image_part", lambda p: None)
    monkeypatch.setattr(answerer, "generate", _fake_generate([], error=RuntimeError("boom")))

    # must not raise
    assert answerer.answer("q", [_page(1)]) == answerer._NOT_FOUND


def test_multiple_regions_pass_through_capped_at_max(monkeypatch):
    monkeypatch.setattr(answerer, "image_part", lambda p: None)
    # Four regions returned; MAX_REGIONS keeps the first few, primary = the first.
    regions = [answerer.Region(source_page=i, box=[i, i, i + 10, i + 10]) for i in range(1, 5)]
    citation = answerer.Citation(answer="two cells", found=True, regions=regions)
    resp = SimpleNamespace(parsed=citation, text="")
    monkeypatch.setattr(answerer, "generate", _fake_generate([], result=resp))

    out = answerer.answer("compare", [_page(i) for i in range(1, 5)])

    assert len(out["regions"]) == answerer.MAX_REGIONS
    assert out["source_page"] == 1 and out["box"] == [1, 1, 11, 11]   # primary = first region
    assert [r["source_page"] for r in out["regions"]] == [1, 2, 3]


def test_not_found_normalizes_regions_to_empty(monkeypatch):
    monkeypatch.setattr(answerer, "image_part", lambda p: None)
    # A found=false response with a stray region -> regions cleared, primary zeroed.
    citation = answerer.Citation(
        answer="not here", found=False,
        regions=[answerer.Region(source_page=1, box=[1, 2, 3, 4])],
    )
    resp = SimpleNamespace(parsed=citation, text="")
    monkeypatch.setattr(answerer, "generate", _fake_generate([], result=resp))

    out = answerer.answer("q", [_page(1)])

    assert out["found"] is False
    assert out["regions"] == [] and out["source_page"] == 0 and out["box"] == []


def test_answer_node_to_highlight_node_survives_bad_response(monkeypatch):
    """A failed answer yields a not-found citation that highlight_node skips cleanly."""
    from src import graph

    monkeypatch.setattr(answerer, "image_part", lambda p: None)
    monkeypatch.setattr(answerer, "generate", _fake_generate([], error=RuntimeError("boom")))

    state = {
        "question": "q",
        "retrieved": [_page(1)],
        "answer": "",
        "citation": None,
        "crop_path": None,
        "annotated_path": None,
    }
    answered = graph.answer_node(state)          # would previously crash on a bad response
    assert answered["citation"]["found"] is False
    state.update(answered)

    assert graph.highlight_node(state) == {
        "crop_path": None,
        "annotated_path": None,
        "cited_regions": [],
        "annotated_paths": [],
    }
