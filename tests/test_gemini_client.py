"""Tests for the pure reliability/usage logic in src.gemini_client.

Covers the retry predicate and the token/cost logging. The network `generate`
path is not exercised here (needs a real API); these guard the custom logic that
decides *whether* to retry and *what* usage to report.
"""

import logging
from types import SimpleNamespace

from google.genai import errors as genai_errors

from src.gemini_client import _is_retryable, _log_usage


class _Timeout(Exception):
    """Stand-in whose name signals a network timeout."""


class _ConnectionReset(Exception):
    """Stand-in whose name signals a dropped connection."""


def test_retryable_api_error_codes():
    assert _is_retryable(genai_errors.APIError(429, {}))   # rate limited
    assert _is_retryable(genai_errors.APIError(503, {}))   # transient server error


def test_non_retryable_api_error_codes():
    assert not _is_retryable(genai_errors.APIError(400, {}))  # bad request
    assert not _is_retryable(genai_errors.APIError(401, {}))  # auth


def test_network_errors_are_retryable_by_name():
    assert _is_retryable(_Timeout())
    assert _is_retryable(_ConnectionReset())


def test_unrelated_errors_are_not_retryable():
    assert not _is_retryable(ValueError("nope"))


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


def test_log_usage_reports_tokens_and_estimated_cost():
    resp = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=1000, candidates_token_count=500, total_token_count=1500))
    records, detach = _capture("gemini")
    try:
        _log_usage(resp, model="gemini-3.5-flash", purpose="answer")
    finally:
        detach()

    assert len(records) == 1
    rec = records[0]
    assert rec.purpose == "answer"
    assert rec.prompt_tokens == 1000
    assert rec.output_tokens == 500
    assert rec.total_tokens == 1500
    # 1000/1e6 * 0.30 + 500/1e6 * 2.50 = 0.0003 + 0.00125
    assert rec.est_cost_usd == round(0.0003 + 0.00125, 6)


def test_log_usage_no_metadata_is_silent():
    records, detach = _capture("gemini")
    try:
        _log_usage(SimpleNamespace(usage_metadata=None), model="x", purpose="answer")
    finally:
        detach()
    assert records == []
