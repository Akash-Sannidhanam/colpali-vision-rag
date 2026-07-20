"""Tests for run_eval's judge call and corpus preflight (eval.run_eval).

Stubs the module's name-as-imported seams (`run_eval.generate`,
`run_eval.list_documents`) per the repo convention - no network, key, or Qdrant.
The run_retrieval_only / run_full orchestration is exercised live in Phase 4's
verification runs, not here.
"""

from types import SimpleNamespace

import pytest

from eval import run_eval
from eval.run_eval import EvalSetupError, JudgeVerdict, check_corpus, judge_answer
from src.config import EVAL_JUDGE_MODEL


def test_judge_answer_routes_through_client_with_judge_model(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
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
    def boom(**kwargs):
        raise RuntimeError("quota")

    monkeypatch.setattr(run_eval, "generate", boom)
    assert judge_answer("q", "a", ["ref"], [{"pdf": "a.pdf", "page": 1}]) is None


def _dataset_row(pdf="a.pdf", page=3, row_id="q1"):
    return {"id": row_id, "question": "q", "gold": [{"pdf": pdf, "page": page}],
            "answer_contains": None, "tags": []}


def test_check_corpus_passes_when_gold_pages_indexed(monkeypatch):
    monkeypatch.setattr(run_eval, "list_documents", lambda: [{"pdf": "a.pdf", "page_count": 5}])
    check_corpus([_dataset_row()])  # no raise


def test_check_corpus_names_missing_pdf(monkeypatch):
    monkeypatch.setattr(run_eval, "list_documents", lambda: [{"pdf": "a.pdf", "page_count": 5}])
    with pytest.raises(EvalSetupError, match="b.pdf"):
        check_corpus([_dataset_row(pdf="b.pdf")])


def test_check_corpus_rejects_gold_page_beyond_page_count(monkeypatch):
    monkeypatch.setattr(run_eval, "list_documents", lambda: [{"pdf": "a.pdf", "page_count": 5}])
    with pytest.raises(EvalSetupError, match="page 9"):
        check_corpus([_dataset_row(page=9)])
