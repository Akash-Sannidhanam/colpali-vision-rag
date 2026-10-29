"""Tests for LangSmith tracing (src.tracing and the runs it adds to the pipeline).

No network: traced runs are captured with a `MagicMock(spec=Client)` handed to
langsmith's `tracing_context`, which is what `traceable` and LangGraph's tracer both
post to. The suite runs with tracing forced off (tests/conftest.py), so everything here
opts in explicitly and nothing leaks into the other tests.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langsmith import Client
from langsmith.run_helpers import tracing_context

from src import gemini_client, tracing


@pytest.fixture(autouse=True)
def _fresh_url_cache(monkeypatch):
    """Each test resolves the project URL from scratch."""
    monkeypatch.setattr(tracing, "_project_url", None)
    monkeypatch.setattr(tracing, "_project_url_failed_at", None)


def _image(nbytes: int, mime: str = "image/png"):
    """A Gemini Part-shaped object carrying inline image bytes."""
    return SimpleNamespace(inline_data=SimpleNamespace(data=b"x" * nbytes, mime_type=mime))


def _response(text: str = '{"ok": true}', usage: bool = True):
    """A generate_content-shaped response: no `.parsed`, raw JSON text, optional usage."""
    meta = SimpleNamespace(prompt_token_count=100, candidates_token_count=20,
                           total_token_count=120) if usage else None
    return SimpleNamespace(parsed=None, text=text, usage_metadata=meta)


def _posted(client: MagicMock, method: str = "create_run") -> list[dict]:
    """The kwargs of every `method` call the mock client received."""
    return [c.kwargs for c in client.method_calls if c[0] == method]


# --- hooks ----------------------------------------------------------------------------

def test_redact_contents_replaces_images_and_keeps_text():
    """Image bytes become a descriptor; the page label beside each image survives."""
    out = tracing.redact_contents({
        "contents": ["PAGE 1 (a.pdf p.3):", _image(3 * 1024, "image/jpeg"), "the prompt"],
        "response_schema": dict,
        "model": "m",
    })
    assert out["contents"] == ["PAGE 1 (a.pdf p.3):", "<image/jpeg 3KB>", "the prompt"]
    assert out["response_schema"] == "dict"
    assert out["model"] == "m"


def test_gemini_outputs_maps_usage_to_langsmith_fields():
    """Token counts use LangSmith's names, which is what makes the run show tokens + cost."""
    out = tracing.gemini_outputs(_response())
    assert out["output"] == '{"ok": true}'
    assert out["usage_metadata"] == {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


def test_gemini_outputs_without_usage_omits_it():
    """A response with no usage_metadata records no token counts rather than zeros."""
    assert "usage_metadata" not in tracing.gemini_outputs(_response(usage=False))


def test_gemini_outputs_prefers_the_parsed_model():
    """When the SDK parsed the schema, the trace shows the parsed object, not raw text."""
    parsed = SimpleNamespace(model_dump=lambda mode: {"answer": "42"})
    out = tracing.gemini_outputs(SimpleNamespace(parsed=parsed, text="ignored", usage_metadata=None))
    assert out["output"] == {"answer": "42"}


def test_retriever_outputs_uses_document_shape():
    """Hits render as LangSmith documents: a readable label plus score metadata."""
    out = tracing.retriever_outputs([{"pdf": "a.pdf", "page_number": 3, "score": 0.9,
                                      "image_path": "/x.png"}])
    assert out == {"documents": [{
        "page_content": "a.pdf p.3", "type": "Document",
        "metadata": {"pdf": "a.pdf", "page_number": 3, "score": 0.9},
    }]}


# --- trace_url ------------------------------------------------------------------------

def test_trace_url_is_none_when_tracing_is_off():
    """Off is the default, and costs no lookup."""
    assert tracing.trace_url(uuid.uuid4().hex) is None


def test_trace_url_links_to_the_dashed_run_id():
    """The request id is uuid4().hex; the URL carries the dashed form LangSmith uses."""
    client = MagicMock(spec=Client)
    client.read_project.return_value = SimpleNamespace(url="https://ls/o/1/projects/p/2")
    rid = uuid.uuid4()
    with tracing_context(enabled=True, client=client):
        url = tracing.trace_url(rid.hex)
    assert url == f"https://ls/o/1/projects/p/2/r/{rid}?poll=true"


def test_trace_url_failure_returns_none_and_backs_off():
    """A failed lookup never raises, and is not retried on the very next query."""
    client = MagicMock(spec=Client)
    client.read_project.side_effect = RuntimeError("401")
    with tracing_context(enabled=True, client=client):
        assert tracing.trace_url(uuid.uuid4().hex) is None
        assert tracing.trace_url(uuid.uuid4().hex) is None
    assert client.read_project.call_count == 1


# --- traced runs ----------------------------------------------------------------------

def test_generate_is_untraced_and_unchanged_when_off(monkeypatch):
    """With tracing off, generate still returns the raw SDK response."""
    resp = _response()
    monkeypatch.setattr(gemini_client, "_client",
                        SimpleNamespace(models=SimpleNamespace(generate_content=lambda **k: resp)))
    assert gemini_client.generate(model="m", contents=["q"], response_schema=dict,
                                  purpose="answer") is resp


def test_generate_records_an_llm_run_without_image_bytes(monkeypatch):
    """One `gemini.<purpose>` llm run, images redacted, tokens and attempts recorded."""
    resp = _response()
    monkeypatch.setattr(gemini_client, "_client",
                        SimpleNamespace(models=SimpleNamespace(generate_content=lambda **k: resp)))
    client = MagicMock(spec=Client)
    with tracing_context(enabled=True, client=client):
        out = gemini_client.generate(model="gemini-x", contents=["PAGE 1:", _image(2048), "q"],
                                     response_schema=dict, purpose="rerank")
    assert out is resp

    (end,) = _posted(client, "update_run")
    assert end["name"] == "gemini.rerank"
    assert end["run_type"] == "llm"
    assert end["inputs"]["contents"] == ["PAGE 1:", "<image/png 2KB>", "q"]
    assert end["outputs"]["usage_metadata"]["total_tokens"] == 120
    meta = end["extra"]["metadata"]
    assert meta["ls_provider"] == "google"
    assert meta["ls_model_name"] == "gemini-x"
    assert meta["attempts"] == 1


def test_run_query_trace_nests_under_a_root_whose_id_is_the_request_id(monkeypatch):
    """Through the real compiled graph: root `rag_query` id == request_id, tagged with
    the source and carrying the config snapshot, with the retriever and both Gemini runs
    inside their nodes."""
    from src import answerer, graph, main, reranker, retrieval

    hits = [{"pdf": "a.pdf", "page_number": i, "score": 1.0 / i, "image_path": "/nope.png"}
            for i in range(1, 6)]
    monkeypatch.setattr(retrieval, "search_multi", lambda qs, weights=None: hits)
    monkeypatch.setattr(retrieval, "embed_query", lambda q: [[0.0]])
    monkeypatch.setattr(answerer, "image_part", lambda p: "IMG")
    monkeypatch.setattr(reranker, "_candidate_part", lambda p: "IMG")
    resp = _response('{"answer": "x", "found": false, "regions": []}', usage=False)
    monkeypatch.setattr(gemini_client, "_client",
                        SimpleNamespace(models=SimpleNamespace(generate_content=lambda **k: resp)))
    monkeypatch.setattr(graph, "_graph", None)
    monkeypatch.setattr(main.tracing, "trace_url", lambda rid: None)  # no project lookup

    client = MagicMock(spec=Client)
    with tracing_context(enabled=True, client=client):
        out = main.run_query("what?", source="eval", trace_metadata={"eval_id": "row-1"})
    from langchain_core.tracers.langchain import wait_for_all_tracers
    wait_for_all_tracers()

    runs = {r["name"]: r for r in _posted(client)}
    root = runs["rag_query"]
    assert root["id"] == uuid.UUID(out["meta"]["request_id"])
    assert "eval" in root["tags"]
    meta = root["extra"]["metadata"]
    assert meta["eval_id"] == "row-1"
    assert meta["retrieve_k"] > 0 and "embed_version" in meta

    def parent(name):
        return runs[name]["parent_run_id"]
    assert parent("retrieval") == runs["retrieve"]["id"]
    assert parent("gemini.rerank") == runs["rerank"]["id"]
    assert parent("gemini.answer") == runs["answer"]["id"]
    assert all(r["trace_id"] == root["id"] for r in runs.values())
