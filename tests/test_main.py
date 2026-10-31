"""Tests for run_query's observability wiring (src.main).

Stubs get_graph so no torch/qdrant/network is touched: a fake graph whose invoke
runs inside the bound request, simulates the two Gemini calls via record_usage, and
returns a canned state. Asserts the request_id is bound during invoke (and threaded
into the invoke config for LangSmith) and that the 'query complete' summary carries
total latency + aggregated tokens.
"""

import logging
import uuid

from src import main, request_context


def _capture(logger_name: str):
    """Attach a list-collecting handler to `logger_name`; returns (records, detach)."""
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    prev = logger.level
    logger.setLevel(logging.INFO)
    return records, lambda: (logger.removeHandler(handler), logger.setLevel(prev))


class _FakeGraph:
    """Stand-in compiled graph: records what it saw, simulates two Gemini calls."""

    def __init__(self, seen: dict):
        """Hold the dict this fake graph records its observations into."""
        self._seen = seen

    def invoke(self, state, config=None):
        """Record the ambient request id and config, then simulate two Gemini calls."""
        self._seen["request_id"] = request_context.current_request_id()
        self._seen["config"] = config
        # simulate the rerank + answer calls feeding the per-query accumulator
        request_context.record_usage(prompt=1000, output=500, total=1500, cost=0.001)
        request_context.record_usage(prompt=200, output=100, total=300, cost=0.0005)
        return {"question": state["question"], "answer": "42"}


def test_run_query_binds_request_id_and_logs_totals(monkeypatch):
    """run_query binds a request id, threads it to the graph config, logs one summary, and unbinds on return."""
    seen: dict = {}
    monkeypatch.setattr(main, "get_graph", lambda: _FakeGraph(seen))

    records, detach = _capture("query")
    try:
        out = main.run_query("what is the revenue?")
    finally:
        detach()

    # request_id was bound (non-default) during invoke and threaded to the config
    assert seen["request_id"] != "-"
    assert seen["config"]["metadata"]["request_id"] == seen["request_id"]
    assert out["answer"] == "42"
    # the request id is also the LangSmith trace id; the CLI is the default source
    assert seen["config"]["run_id"] == uuid.UUID(seen["request_id"])
    assert seen["config"]["tags"] == ["cli"]
    # tracing is off in the suite, so there is no trace to link to
    assert out["meta"]["trace_url"] is None

    # exactly one 'query complete' summary with total latency + summed tokens
    complete = [r for r in records if r.getMessage() == "query complete"]
    assert len(complete) == 1
    rec = complete[0]
    assert rec.total_tokens == 1800
    assert rec.gemini_calls == 2
    assert rec.latency_ms >= 0

    # scope is cleaned up after the call returns
    assert request_context.current_request_id() == "-"


def test_run_query_threads_source_and_trace_metadata(monkeypatch):
    """`source` becomes the trace tag and `trace_metadata` joins the config snapshot."""
    seen: dict = {}
    monkeypatch.setattr(main, "get_graph", lambda: _FakeGraph(seen))

    main.run_query("q", source="eval", trace_metadata={"eval_id": "row-1"})

    config = seen["config"]
    assert config["tags"] == ["eval"]
    assert config["metadata"]["eval_id"] == "row-1"
    assert config["metadata"]["rerank_k"] == main.tracing.run_metadata()["rerank_k"]
