"""Tests for the structured log formatter (src.logging_setup._Formatter).

Pure formatting logic - no I/O, no config side effects beyond building records.
"""

import json
import logging

from src.logging_setup import _Formatter


def _record(**extra) -> logging.LogRecord:
    """Build a LogRecord carrying arbitrary structured `extra` fields."""
    base = {"name": "test", "levelname": "INFO", "levelno": logging.INFO, "msg": "hello"}
    return logging.makeLogRecord({**base, **extra})


def test_human_format_includes_message_and_extras():
    line = _Formatter(json_mode=False).format(_record(purpose="answer", total_tokens=42))
    assert "hello" in line
    assert "purpose=answer" in line
    assert "total_tokens=42" in line


def test_json_format_is_parseable_with_extras():
    out = _Formatter(json_mode=True).format(_record(purpose="rerank", total_tokens=7))
    payload = json.loads(out)  # one JSON object per line
    assert payload["msg"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["purpose"] == "rerank"
    assert payload["total_tokens"] == 7


def test_reserved_record_fields_are_not_leaked_as_extras():
    # Standard LogRecord attributes must not appear as structured fields.
    payload = json.loads(_Formatter(json_mode=True).format(_record()))
    for reserved in ("args", "levelno", "pathname", "msecs", "processName"):
        assert reserved not in payload
