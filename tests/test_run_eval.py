"""Tests for run_eval's judge call and corpus preflight (eval.run_eval).

Stubs the module's name-as-imported seams (`run_eval.generate`,
`run_eval.list_documents`) and the pipeline entry points the run_* functions import
lazily (`src.main.run_query`, `src.embedder.embed_query`, `src.vector_store.search`)
per the repo convention - no network, key, or Qdrant. The orchestration tests cover
the answerable-vs-unanswerable *row shapes*, which is where a scoring bug would hide;
the live numbers still come from a real eval run.
"""

import json
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
    assert len(rows) >= 83

    negatives = [r for r in rows if r["unanswerable"]]
    cross_doc = [r for r in rows if len({g["pdf"] for g in r["gold"]}) > 1]
    assert len(negatives) >= 10, "no abstention rows left - hallucination rate goes unmeasured"
    # 20, not 6: gold_coverage_avg is the only metric with real headroom, and at n=6 one
    # question moved it 0.083 - wider than the run-to-run variance measured on identical
    # code. Trimming this slice back doesn't just lose rows, it makes the metric unable
    # to adjudicate the RERANK_K question it exists for.
    assert len(cross_doc) >= 20, "cross-document slice too small to resolve gold_coverage_avg"

    assert all("unanswerable" in r["tags"] for r in negatives)

    # A cross-document row must never carry a bare any-of label: one hit on either
    # document's fact would satisfy it, which is how substring_accuracy scored 1.0 on
    # this slice while the judge was failing rows on it. All-of, or nothing at all.
    for r in cross_doc:
        assert not r["answer_contains"], (
            f"{r['id']}: any-of `answer_contains` can pass on half a cross-document "
            f"answer - use `answer_contains_all`, or drop it and let the judge score it"
        )


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
            "answer_contains": None, "answer_contains_all": None, "tags": [],
            "unanswerable": False}


def _unanswerable_row(row_id="n1"):
    """One dataset row whose answer is nowhere in the corpus."""
    return {"id": row_id, "question": "q", "gold": [], "answer_contains": None, "answer_contains_all": None,
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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_parse_gates_rejects_a_non_finite_legacy_threshold(value):
    """--fail-under-recall reaches the same gate list and needs the same check.

    argparse's `type=float` accepts "nan", and this path skipped the validation the
    --gate specs get - guarding one of two entry points to one list is no guard.
    """
    with pytest.raises(ValueError, match="finite"):
        parse_gates(None, "recall@1", value)


@pytest.mark.parametrize("spec", ["recall@1:nan", "recall@1:inf", "recall@1:-inf"])
def test_parse_gates_rejects_non_finite_thresholds(spec):
    """float() parses "nan", and `value < nan` is always False - a gate that can't fire.

    Silently un-fireable is the one failure mode the fail-closed design exists to
    prevent, so it has to be rejected at parse time.
    """
    with pytest.raises(ValueError, match="finite"):
        parse_gates([spec], "recall@10", None)


# --- the degraded-run guard ---

def _scored_row(row_id="q1", degraded=0, judge=..., unanswerable=False):
    """One run_full output row with a given per-row degraded-call count."""
    row = {"id": row_id, "tags": [], "meta": {"degraded_calls": degraded, "gemini_calls": 2}}
    if unanswerable:
        row["abstention_correct"] = True
    elif judge is not ...:
        row["judge"] = judge
    return row


def test_degradation_summary_of_a_clean_run_is_all_zeros():
    """A healthy run records zeros - positive evidence the guard ran at all."""
    rows = [_scored_row("q1"), _scored_row("q2")]
    assert run_eval.degradation_summary(rows, use_judge=False) == {
        "degraded_calls": 0, "rows_with_degraded": 0, "judge_na": 0, "degraded_row_frac": 0.0,
    }


def test_degradation_summary_counts_calls_and_rows_separately():
    """Two failures on one row is 2 calls but 1 affected row - the fraction uses rows."""
    rows = [_scored_row("q1", degraded=2), _scored_row("q2"), _scored_row("q3"), _scored_row("q4")]
    out = run_eval.degradation_summary(rows, use_judge=False)
    assert out["degraded_calls"] == 2
    assert out["rows_with_degraded"] == 1
    assert out["degraded_row_frac"] == 0.25


def test_degradation_summary_counts_judge_na_as_degradation():
    """A None judge verdict IS a failed judge call - no separate counter needed."""
    rows = [_scored_row("q1", judge=None), _scored_row("q2", judge={"correct": True, "score": 5})]
    out = run_eval.degradation_summary(rows, use_judge=True)
    assert out["judge_na"] == 1
    assert out["rows_with_degraded"] == 1        # the judge-N/A row counts as affected


def test_degradation_summary_ignores_judge_when_not_requested():
    """Without --judge every row is legitimately judge-less; that isn't degradation."""
    rows = [_scored_row("q1"), _scored_row("q2")]
    assert run_eval.degradation_summary(rows, use_judge=False)["judge_na"] == 0


def test_degradation_summary_skips_the_judge_on_unanswerable_rows():
    """run_full deliberately never judges a negative row, so its absence isn't a failure."""
    rows = [_scored_row("n1", unanswerable=True), _scored_row("q1", judge={"correct": True})]
    assert run_eval.degradation_summary(rows, use_judge=True)["judge_na"] == 0


def test_degradation_summary_handles_rows_without_meta():
    """Retrieval-only rows carry no meta and touch no Gemini - structurally zero."""
    rows = [{"id": "q1", "tags": [], "gold_rank": 1}, {"id": "n1", "tags": [], "unanswerable": True}]
    assert run_eval.degradation_summary(rows, use_judge=False)["degraded_calls"] == 0


def test_degradation_summary_of_an_empty_run_does_not_divide_by_zero():
    """No rows means no fraction to compute, not a ZeroDivisionError."""
    assert run_eval.degradation_summary([], use_judge=False)["degraded_row_frac"] == 0.0


@pytest.mark.parametrize(
    "frac,limit,expected",
    [(0.0, 0.02, False), (0.02, 0.02, False), (0.021, 0.02, True), (1.0, 0.02, True)],
)
def test_degradation_breached_is_strictly_above_the_limit(frac, limit, expected):
    """At the limit is tolerated; above it is not. A run of all-degraded rows always trips."""
    assert run_eval.degradation_breached({"degraded_row_frac": frac}, limit) is expected


def test_degradation_breach_is_the_outage_that_flattered_abstention():
    """The regression this guard exists for.

    A full run against depleted quota degrades every answer to a not-found citation.
    `abstention_correct` is `not citation["found"]`, so every negative row scores
    *correct* and abstention_accuracy reads 1.0 - the most dangerous number in the
    report to falsely peg. The row-level counts are what make it visible.
    """
    rows = [_scored_row(f"q{i}", degraded=1) for i in range(10)]
    out = run_eval.degradation_summary(rows, use_judge=False)
    assert out["degraded_row_frac"] == 1.0
    assert run_eval.degradation_breached(out, 0.02) is True


def _main_with_rows(monkeypatch, tmp_path, rows, argv=()):
    """Drive run_eval.main() end-to-end over canned rows, stubbing every live seam."""
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(
        '{"id": "q1", "question": "q", "gold": [{"pdf": "a.pdf", "page": 1}]}\n'
    )
    monkeypatch.setattr(run_eval, "ping", lambda: None)
    monkeypatch.setattr(run_eval, "check_corpus", lambda d: None)
    monkeypatch.setattr(run_eval, "close_client", lambda: None)
    monkeypatch.setattr(run_eval.config, "validate", lambda: None)
    monkeypatch.setattr(run_eval, "run_full", lambda d, j: rows)
    monkeypatch.setattr(run_eval, "DEFAULT_REPORT_DIR", tmp_path / "reports")
    code = run_eval.main(["--dataset", str(dataset), *argv])
    written = sorted((tmp_path / "reports").glob("*.json"))
    return code, written


def test_main_refuses_to_pin_a_degraded_run(monkeypatch, tmp_path):
    """The whole point: an outage exits 2 and writes a report nobody can mistake for a baseline."""
    rows = [_scored_row(f"q{i}", degraded=1) for i in range(5)]
    code, written = _main_with_rows(monkeypatch, tmp_path, rows, ["--gate", "recall@1:0.99"])

    assert code == 2                                  # not 0, and not the gate's 1
    (report_path,) = written
    assert report_path.name.startswith("degraded_")   # never eval_*, which is what gets pinned
    report = json.loads(report_path.read_text())
    assert report["degraded_run"] is True
    assert report["degradation"]["rows_with_degraded"] == 5


def test_main_writes_a_normal_report_when_nothing_degraded(monkeypatch, tmp_path):
    """A clean run is unaffected, and records zeros as evidence the guard ran."""
    code, written = _main_with_rows(monkeypatch, tmp_path, [_scored_row("q1")])

    assert code == 0
    (report_path,) = written
    assert report_path.name.startswith("eval_")
    report = json.loads(report_path.read_text())
    assert report["degraded_run"] is False
    assert report["degradation"]["degraded_calls"] == 0


def test_main_tolerates_a_single_transient_failure(monkeypatch, tmp_path):
    """One degraded row in 69 is under the default limit - a 429 shouldn't void a run."""
    rows = [_scored_row("q0", degraded=1)] + [_scored_row(f"q{i}") for i in range(1, 69)]
    code, written = _main_with_rows(monkeypatch, tmp_path, rows)

    assert code == 0
    assert written[0].name.startswith("eval_")


def test_main_honors_an_explicit_output_path_even_when_degraded(monkeypatch, tmp_path):
    """--output is obeyed rather than silently renamed; the stamp still marks the run."""
    out = tmp_path / "reports" / "mine.json"
    out.parent.mkdir(parents=True)
    rows = [_scored_row(f"q{i}", degraded=1) for i in range(5)]
    code, _ = _main_with_rows(monkeypatch, tmp_path, rows, ["--output", str(out)])

    assert code == 2
    assert json.loads(out.read_text())["degraded_run"] is True


def test_config_snapshot_stores_a_repo_relative_dataset_path():
    """An absolute default path would bake the operator's home dir into the baseline."""
    snapshot = run_eval._config_snapshot("full", str(DEFAULT_DATASET), use_judge=False)
    assert snapshot["dataset"] == "eval/dataset.jsonl"


def test_config_snapshot_leaves_an_outside_path_alone():
    """A dataset deliberately kept outside the repo still records where it came from."""
    snapshot = run_eval._config_snapshot("full", "/somewhere/else/custom.jsonl", use_judge=False)
    assert snapshot["dataset"] == "/somewhere/else/custom.jsonl"
