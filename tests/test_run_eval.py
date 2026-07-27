"""Tests for run_eval's judge call and corpus preflight (eval.run_eval).

Stubs the module's name-as-imported seams (`run_eval.generate`,
`run_eval.list_documents`) and the pipeline entry points the run_* functions import
lazily (`src.main.run_query`, `src.embedder.embed_query`, `src.vector_store.search`)
per the repo convention - no network, key, or Qdrant. The orchestration tests cover
the answerable-vs-unanswerable *row shapes*, which is where a scoring bug would hide;
the live numbers still come from a real eval run.
"""

from types import SimpleNamespace

import pytest

from eval import run_eval
from eval.run_eval import (
    DEFAULT_DATASET,
    EvalSetupError,
    JudgeVerdict,
    check_corpus,
    gate_status,
    judge_answer,
    parse_gates,
)
from eval.scoring import load_dataset
from src.config import EVAL_JUDGE_MODEL

# --- the shipped dataset itself ---

def test_shipped_dataset_parses_and_holds_its_invariants():
    """The real eval/dataset.jsonl is valid and still covers every question kind.

    Reading one tracked file is worth the exception to the no-I/O convention: a
    malformed row otherwise surfaces only as exit 2 partway through a live eval run,
    after the models are loaded.
    """
    rows = load_dataset(DEFAULT_DATASET.read_text().splitlines())
    assert len(rows) >= 69

    negatives = [r for r in rows if r["unanswerable"]]
    cross_doc = [r for r in rows if len({g["pdf"] for g in r["gold"]}) > 1]
    assert len(negatives) >= 10, "no abstention rows left - hallucination rate goes unmeasured"
    assert len(cross_doc) >= 6, "no cross-document rows left - RERANK_K goes unpressured"

    # An answerable row without an expected substring scores nothing but recall.
    assert all(r["answer_contains"] for r in rows if not r["unanswerable"])
    assert all("unanswerable" in r["tags"] for r in negatives)


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
            "answer_contains": None, "tags": [], "unanswerable": False}


def _unanswerable_row(row_id="n1"):
    """One dataset row whose answer is nowhere in the corpus."""
    return {"id": row_id, "question": "q", "gold": [], "answer_contains": None,
            "tags": ["unanswerable"], "unanswerable": True}


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


def test_check_corpus_ignores_unanswerable_rows(monkeypatch):
    """A row with no gold page has nothing to preflight and must not trip the check."""
    monkeypatch.setattr(run_eval, "list_documents", lambda: [{"pdf": "a.pdf", "page_count": 5}])
    check_corpus([_unanswerable_row()])  # no raise


# --- row shapes: answerable vs unanswerable questions ---

def test_run_retrieval_only_does_not_score_unanswerable_rows(monkeypatch):
    """A negative question is kept in `n` but carries no gold_rank to be missed on.

    Retrieval-only mode has no citation, so there is nothing to score; emitting
    gold_rank=None would count a correctly-unanswerable question as a recall miss.
    """
    searched = []
    monkeypatch.setattr("src.embedder.embed_query", lambda q: [[0.1]])
    monkeypatch.setattr("src.vector_store.search", lambda mv: searched.append(mv) or [
        {"pdf": "a.pdf", "page_number": 3, "image_path": "p3.png", "score": 0.9},
    ])

    rows = run_eval.run_retrieval_only([_dataset_row(), _unanswerable_row()])

    assert [r["id"] for r in rows] == ["q1", "n1"]
    assert rows[0]["gold_rank"] == 1
    assert "gold_rank" not in rows[1]
    assert len(searched) == 1  # the negative row never reached the retriever


def _stub_run_query(monkeypatch, *, found: bool, confidence: str = "high"):
    """Stub main.run_query with one reranked page and a citation pointing at it."""
    reranked = [{"pdf": "a.pdf", "page_number": 3, "image_path": "p3.png", "score": 0.9}]
    citation = {"answer": "180", "found": found, "confidence": confidence,
                "source_page": 1 if found else 0, "box": []}
    monkeypatch.setattr("src.main.run_query", lambda q: {
        "answer": "180", "citation": citation, "retrieved": reranked, "candidates": reranked,
        "meta": {"latency_ms": 12.0, "retrieval_confidence": 0.81},
    })


def test_run_full_scores_an_unanswerable_row_on_abstention_alone(monkeypatch):
    """Negative rows omit every retrieval/citation key so they can't enter a denominator."""
    _stub_run_query(monkeypatch, found=False)
    monkeypatch.setattr(run_eval, "judge_answer", lambda *a: pytest.fail("judge ran on a negative"))

    (row,) = run_eval.run_full([_unanswerable_row()], use_judge=True)

    assert row["abstention_correct"] is True
    assert row["found"] is False
    for absent in ("gold_rank", "rerank_hit", "citation_correct", "substring_match", "judge"):
        assert absent not in row


def test_run_full_counts_an_answered_unanswerable_question_as_a_hallucination(monkeypatch):
    """Answering a question the corpus can't support is the failure this metric exists for."""
    _stub_run_query(monkeypatch, found=True)
    (row,) = run_eval.run_full([_unanswerable_row()], use_judge=False)
    assert row["abstention_correct"] is False


def test_run_full_carries_coverage_and_both_confidence_signals(monkeypatch):
    """Answerable rows pick up gold_doc_coverage plus the two confidence values."""
    _stub_run_query(monkeypatch, found=True, confidence="medium")

    (row,) = run_eval.run_full([_dataset_row()], use_judge=False)

    assert row["citation_correct"] is True
    assert row["gold_doc_coverage"] is None       # single-document gold: N/A
    assert row["retrieval_confidence"] == 0.81
    assert row["self_confidence"] == "medium"
    assert "abstention_correct" not in row


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


# --- --gate METRIC:MIN parsing ---

def test_parse_gates_collects_repeated_specs():
    """Several --gate flags become several (metric, minimum) pairs."""
    gates = parse_gates(["recall@1:0.70", "gold_coverage_avg:0.60"], "recall@10", None)
    assert gates == [("recall@1", 0.7), ("gold_coverage_avg", 0.6)]


def test_parse_gates_splits_on_the_last_colon():
    """Metric names contain no colon, but splitting from the right is the safe choice."""
    assert parse_gates(["recall@1:0.7"], "recall@10", None) == [("recall@1", 0.7)]


def test_parse_gates_appends_the_legacy_flag_pair():
    """--fail-metric/--fail-under-recall still work, alongside any --gate flags."""
    gates = parse_gates(["recall@1:0.70"], "citation_accuracy", 0.9)
    assert gates == [("recall@1", 0.7), ("citation_accuracy", 0.9)]


def test_parse_gates_without_any_threshold_is_empty():
    """No gate flags means nothing to fail on."""
    assert parse_gates(None, "recall@10", None) == []


@pytest.mark.parametrize("spec", ["recall@1", "recall@1:high", ":0.7"])
def test_parse_gates_rejects_a_malformed_spec(spec):
    """A typo fails at parse time, not after a multi-minute run has already burned."""
    with pytest.raises(ValueError, match="gate"):
        parse_gates([spec], "recall@10", None)
