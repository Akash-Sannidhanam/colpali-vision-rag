"""Tests for run_eval's judge call and corpus preflight (eval.run_eval).

Stubs the module's name-as-imported seams (`run_eval.generate`,
`run_eval.list_documents`) per the repo convention - no network, key, or Qdrant.
The run_retrieval_only / run_full orchestration is exercised live in Phase 4's
verification runs, not here.
"""

from types import SimpleNamespace

import pytest

from eval import run_eval
from eval.run_eval import EvalSetupError, JudgeVerdict, check_corpus, gate_status, judge_answer
from src.config import EVAL_JUDGE_MODEL


def test_judge_answer_routes_through_client_with_judge_model(monkeypatch):
    """The judge goes through the shared client tagged with EVAL_JUDGE_MODEL and purpose=judge."""
    captured = {}

    def fake_generate(**kwargs):
        """A stubbed Gemini call returning a fixed judge verdict."""
        captured.update(kwargs)
        return SimpleNamespace(
            parsed=JudgeVerdict(correct=True, score=5, reasoning="matches the reference"),
            text="",
        )

    monkeypatch.setattr(run_eval, "generate", fake_generate)
    verdict = judge_answer("What was Q4 revenue?", "It was 180 thousand.", ["180"], [{"pdf": "a.pdf", "page": 1}])

    assert verdict == {"correct": True, "score": 5, "reasoning": "matches the reference"}
    assert captured["purpose"] == "judge"
    assert captured["model"] == EVAL_JUDGE_MODEL
    assert captured["response_schema"] is JudgeVerdict


def test_judge_answer_failure_returns_none_not_raise(monkeypatch):
    """A judge outage degrades to N/A instead of failing the whole run."""
    def boom(**kwargs):
        """Raise, to exercise the judge's degradation path."""
        raise RuntimeError("quota")

    monkeypatch.setattr(run_eval, "generate", boom)
    assert judge_answer("q", "a", ["ref"], [{"pdf": "a.pdf", "page": 1}]) is None


def _dataset_row(pdf="a.pdf", page=3, row_id="q1"):
    """One labeled dataset row pointing at a single gold page."""
    return {"id": row_id, "question": "q", "gold": [{"pdf": pdf, "page": page}],
            "answer_contains": None, "tags": []}


def test_check_corpus_passes_when_gold_pages_indexed(monkeypatch):
    """A dataset whose gold pages are all indexed passes preflight."""
    monkeypatch.setattr(run_eval, "list_documents", lambda: [{"pdf": "a.pdf", "page_count": 5}])
    check_corpus([_dataset_row()])  # no raise


def test_check_corpus_names_missing_pdf(monkeypatch):
    """Preflight fails naming the gold document that isn't indexed."""
    monkeypatch.setattr(run_eval, "list_documents", lambda: [{"pdf": "a.pdf", "page_count": 5}])
    with pytest.raises(EvalSetupError, match="b.pdf"):
        check_corpus([_dataset_row(pdf="b.pdf")])


def test_check_corpus_rejects_gold_page_beyond_page_count(monkeypatch):
    """Preflight fails naming a gold page beyond the document's length."""
    monkeypatch.setattr(run_eval, "list_documents", lambda: [{"pdf": "a.pdf", "page_count": 5}])
    with pytest.raises(EvalSetupError, match="page 9"):
        check_corpus([_dataset_row(page=9)])


# --- the --fail-metric / --fail-under-recall gate ---

_SUMMARY = {"recall@1": 0.77, "recall@3": 0.95, "recall@10": 1.0, "citation_accuracy": 1.0}


def test_gate_no_threshold_never_fails():
    """With no threshold configured the CI gate never fails the run."""
    assert gate_status(_SUMMARY, "recall@1", None) == (False, None)


def test_gate_fails_when_chosen_metric_below_threshold():
    """A metric under its threshold trips the gate and reports the value."""
    # Gating on recall@1 (which has headroom) catches a regression the saturated
    # recall@10 default would miss.
    assert gate_status(_SUMMARY, "recall@1", 0.9) == (True, 0.77)


def test_gate_passes_when_metric_meets_threshold():
    """A metric at or above its threshold passes."""
    assert gate_status(_SUMMARY, "recall@3", 0.9) == (False, 0.95)


def test_gate_fails_on_unknown_or_na_metric():
    """An unknown or N/A metric fails closed rather than passing silently."""
    # A typo'd or N/A metric can't silently pass the gate.
    failed, value = gate_status(_SUMMARY, "recall@2", 0.5)
    assert failed is True and value is None
