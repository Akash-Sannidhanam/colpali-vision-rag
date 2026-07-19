"""Tests for the per-node timing wrapper (src.graph._timed).

Pure logic: wraps a trivial function and asserts the start/end + latency_ms log
lines. No graph compile, no langgraph invoke, no models or network.
"""

import logging

from src import graph


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


def test_timed_logs_start_and_end_with_node_and_latency():
    records, detach = _capture("graph")
    try:
        wrapped = graph._timed("retrieve", lambda state: {"ok": True})
        out = wrapped({"question": "q"})
    finally:
        detach()

    assert out == {"ok": True}                       # return value passes through
    assert len(records) == 2
    assert records[0].getMessage() == "node start" and records[0].node == "retrieve"
    assert records[1].getMessage() == "node end" and records[1].node == "retrieve"
    assert records[1].latency_ms >= 0


def test_timed_logs_end_even_when_node_raises():
    def boom(state):
        raise RuntimeError("nope")

    records, detach = _capture("graph")
    raised = False
    try:
        try:
            graph._timed("answer", boom)({})
        except RuntimeError:
            raised = True
    finally:
        detach()

    assert raised                                    # the exception still propagates
    assert [r.getMessage() for r in records] == ["node start", "node end"]
    assert records[1].node == "answer" and records[1].latency_ms >= 0
