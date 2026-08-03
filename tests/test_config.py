"""Tests for `config.validate()`, the startup fail-fast check.

Patch the by-value module globals on `src.config` itself - they are read inside
`validate()` at call time, so monkeypatching them is what steers the branch.
"""

import math

import pytest

from src import config


def _ok(monkeypatch):
    """Put config in a state where validate() passes, so a test can break one thing."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "CANDIDATE_FANOUT", 2.0)


def test_validate_passes_on_a_sane_config(monkeypatch):
    """The happy path returns None rather than raising."""
    _ok(monkeypatch)
    assert config.validate() is None


def test_validate_rejects_a_missing_api_key(monkeypatch):
    """An empty key is caught at startup, not at the first Gemini call."""
    _ok(monkeypatch)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        config.validate()


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_validate_rejects_a_non_finite_fanout(monkeypatch, bad):
    """float() parses "inf"/"nan" happily, and round() then raises on *every* search.

    Catching it at startup turns a per-query OverflowError/ValueError deep inside
    `vector_store.search` into one clear message naming the knob.
    """
    _ok(monkeypatch)
    monkeypatch.setattr(config, "CANDIDATE_FANOUT", bad)

    with pytest.raises(RuntimeError, match="CANDIDATE_FANOUT"):
        config.validate()
