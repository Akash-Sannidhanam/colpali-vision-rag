"""Tests for the structured log formatter (src.logging_setup._Formatter).

Pure formatting logic - no I/O, no config side effects beyond building records.
"""

import json
import logging

from src import request_context
from src.logging_setup import _Formatter, _RequestIdFilter


def _record(**extra) -> logging.LogRecord:
    """Build a LogRecord carrying arbitrary structured `extra` fields."""
    base = {"name": "test", "levelname": "INFO", "levelno": logging.INFO, "msg": "hello"}
    return logging.makeLogRecord({**base, **extra})


def test_human_format_includes_message_and_extras():
    """Human lines carry the message plus every `extra=` field as key=value."""
    line = _Formatter(json_mode=False).format(_record(purpose="answer", total_tokens=42))
    assert "hello" in line
    assert "purpose=answer" in line
    assert "total_tokens=42" in line


def test_json_format_is_parseable_with_extras():
    """JSON mode emits one parseable object per line, extras included."""
    out = _Formatter(json_mode=True).format(_record(purpose="rerank", total_tokens=7))
    payload = json.loads(out)  # one JSON object per line
    assert payload["msg"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["purpose"] == "rerank"
    assert payload["total_tokens"] == 7


def test_reserved_record_fields_are_not_leaked_as_extras():
    """Stdlib LogRecord attributes stay out of the rendered extras."""
    # Standard LogRecord attributes must not appear as structured fields.
    payload = json.loads(_Formatter(json_mode=True).format(_record()))
    for reserved in ("args", "levelno", "pathname", "msecs", "processName"):
        assert reserved not in payload


def test_request_id_filter_stamps_the_bound_id():
    """Inside a request the filter stamps the bound id onto the record."""
    record = _record()
    scope = request_context.begin_request()
    try:
        assert _RequestIdFilter().filter(record) is True
        assert record.request_id == request_context.current_request_id()
        assert record.request_id != "-"
    finally:
        request_context.end_request(scope)


def test_request_id_filter_defaults_outside_a_request():
    """Outside a request the id renders as '-' rather than failing."""
    record = _record()
    _RequestIdFilter().filter(record)
    assert record.request_id == "-"


def test_request_id_renders_as_a_field():
    """request_id isn't reserved, so the formatter emits it like any other extra."""
    payload = json.loads(_Formatter(json_mode=True).format(_record(request_id="abc123")))
    assert payload["request_id"] == "abc123"
