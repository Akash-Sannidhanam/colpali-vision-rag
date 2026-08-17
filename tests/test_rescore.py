"""Tests for the offline re-scorer (eval.rescore).

Pure logic, like the rest of the eval tests: a report is a dict and a dataset is a list
of dicts, so nothing here needs Qdrant, a model, or a key.

The load-bearing property is the **round trip** - re-scoring against unchanged labels
must reproduce the source summary exactly. Everything else the tool does is only
trustworthy if that holds, because a rescore that silently drifts would present its own
drift as a label-change effect. It is asserted twice: once on a synthetic report, and
once against the committed baseline, which is the artifact people will actually point
this at.
"""

import json

import pytest

from eval import rescore
from eval.rescore import build_report, rescore_rows


def _dataset_row(row_id="q1", gold=None, **extra):
    return {
        "id": row_id,
        "question": "q?",
        "gold": gold if gold is not None else [{"pdf": "a.pdf", "page": 1}],
        "answer_contains": None,
        "answer_contains_all": None,
        "unanswerable": False,
        "tags": [],
        **extra,
    }


def _report_row(row_id="q1", *, candidates, reranked, cited=None, found=True, answer=""):
    return {
        "id": row_id,
        "tags": [],
        "found": found,
        "answer": answer,
        "cited": cited,
        "gold_rank": None,
        "rerank_hit": False,
        "citation_correct": False,
        "candidate_doc_coverage": None,
        "gold_doc_coverage": None,
        "candidate_pages": [{"pdf": p, "page": n} for p, n in candidates],
        "reranked_pages": [{"pdf": p, "page": n} for p, n in reranked],
        "substring_match": None,
    }


# --- the round trip, which is the whole guarantee ---

def test_unchanged_labels_reproduce_every_row(tmp_path):
    """Re-scoring with the same labels must not move a single value."""
    rows = [
        _report_row("q1", candidates=[("a.pdf", 1), ("b.pdf", 2)], reranked=[("a.pdf", 1)],
                    cited={"pdf": "a.pdf", "page": 1}),
    ]
    # Score once to get the "measured" values, then score that output again.
    first, _, _ = rescore_rows(rows, [_dataset_row("q1")])
    second, changes, _ = rescore_rows(first, [_dataset_row("q1")])
    assert first == second
    assert changes == []


def test_committed_baseline_round_trips_on_the_metrics_it_owns():
    """The real artifact: eval/reports/calib_baseline.json re-scores to itself.

    Pinned on `calib_baseline` rather than `baseline_decomposed` deliberately. The
    latter predates the confidence-calibration pass, so its stored summary was written
    by superseded calibration code and cannot be reproduced by the current scorer -
    a real (small) fact about that report, and not something this tool can fix.
    """
    report = json.loads(
        (rescore.Path(__file__).resolve().parent.parent
         / "eval/reports/calib_baseline.json").read_text()
    )
    dataset_text = (rescore.Path(__file__).resolve().parent.parent
                    / "eval/dataset.jsonl").read_text()
    from eval.scoring import load_dataset
    rows, changes, orphans = rescore_rows(report["rows"], load_dataset(dataset_text.splitlines()))
    # The dataset has since been relabelled, so rows *will* change; what must hold is
    # that every row the labels did not touch is byte-identical.
    changed = {c["id"] for c in changes} | set(orphans)
    for before, after in zip(report["rows"], rows):
        if before["id"] not in changed:
            assert before == after, f"{before['id']} drifted with no label change"


# --- what it recomputes ---

def test_a_widened_gold_lifts_coverage_and_is_reported():
    """Adding the second gold document's page to the labels moves coverage 0.5 -> 1.0."""
    rows = [_report_row("q1", candidates=[("a.pdf", 1), ("b.pdf", 7)],
                        reranked=[("a.pdf", 1), ("b.pdf", 7)])]
    narrow = [_dataset_row("q1", gold=[{"pdf": "a.pdf", "page": 1}, {"pdf": "b.pdf", "page": 9}])]
    wide = [_dataset_row("q1", gold=[{"pdf": "a.pdf", "page": 1}, {"pdf": "b.pdf", "page": 7}])]

    before, _, _ = rescore_rows(rows, narrow)
    assert before[0]["gold_doc_coverage"] == 0.5

    after, changes, _ = rescore_rows(rows, wide)
    assert after[0]["gold_doc_coverage"] == 1.0
    assert changes[0]["id"] == "q1"
    assert "gold_doc_coverage" in changes[0]["moved"]


def test_citation_is_rescored_from_the_stored_cited_page():
    """`cited` is the stored resolution of source_page, so it is what gets re-judged."""
    rows = [_report_row("q1", candidates=[("a.pdf", 3)], reranked=[("a.pdf", 3)],
                        cited={"pdf": "a.pdf", "page": 3})]
    hit, _, _ = rescore_rows(rows, [_dataset_row("q1", gold=[{"pdf": "a.pdf", "page": 3}])])
    miss, _, _ = rescore_rows(rows, [_dataset_row("q1", gold=[{"pdf": "a.pdf", "page": 4}])])
    assert hit[0]["citation_correct"] is True
    assert miss[0]["citation_correct"] is False


def test_a_not_found_row_never_scores_a_correct_citation():
    rows = [_report_row("q1", candidates=[("a.pdf", 1)], reranked=[("a.pdf", 1)],
                        cited=None, found=False)]
    out, _, _ = rescore_rows(rows, [_dataset_row("q1", gold=[{"pdf": "a.pdf", "page": 1}])])
    assert out[0]["citation_correct"] is False


def test_unanswerable_rows_are_left_alone():
    """Their shape carries abstention_correct only; inventing the other keys would put a
    correctly-declined question into a retrieval denominator."""
    row = {"id": "n1", "tags": [], "found": False, "abstention_correct": True, "answer": ""}
    out, changes, _ = rescore_rows([row], [_dataset_row("n1", unanswerable=True, gold=[])])
    assert out == [row]
    assert changes == []


def test_retrieval_only_rows_do_not_grow_a_rerank_that_never_ran():
    row = {
        "id": "q1", "tags": [], "gold_rank": None, "candidate_doc_coverage": None,
        "candidate_pages": [{"pdf": "a.pdf", "page": 1}],
    }
    out, _, _ = rescore_rows([row], [_dataset_row("q1")])
    assert out[0]["gold_rank"] == 1
    assert "gold_doc_coverage" not in out[0]
    assert "rerank_hit" not in out[0]


def test_rows_missing_from_the_dataset_are_kept_and_named():
    rows = [_report_row("gone", candidates=[("a.pdf", 1)], reranked=[("a.pdf", 1)])]
    out, changes, orphans = rescore_rows(rows, [_dataset_row("other")])
    assert orphans == ["gone"]
    assert out == rows and changes == []


# --- provenance and the judge caveat ---

def test_a_changed_row_with_a_judge_verdict_is_flagged_stale():
    """The judge prompt embeds the gold labels, so a relabelled row's verdict is stale.

    It cannot be recomputed offline, so the only honest move is to say so - otherwise
    judge_accuracy silently mixes two label sets.
    """
    row = _report_row("q1", candidates=[("a.pdf", 1), ("b.pdf", 7)],
                      reranked=[("a.pdf", 1), ("b.pdf", 7)])
    row["judge"] = {"correct": True, "score": 5, "reasoning": "..."}
    dataset = [_dataset_row("q1", gold=[{"pdf": "a.pdf", "page": 1}, {"pdf": "b.pdf", "page": 7}])]
    rows, changes, _ = rescore_rows([row], dataset)
    report = build_report({"summary": {}, "config": {"retrieve_k": 12}}, rescore.Path("x.json"),
                          rows, changes)
    assert report["rescored"]["stale_judge_rows"] == ["q1"]


def test_report_records_where_it_came_from():
    rows = [_report_row("q1", candidates=[("a.pdf", 1)], reranked=[("a.pdf", 1)])]
    scored, changes, _ = rescore_rows(rows, [_dataset_row("q1")])
    report = build_report({"summary": {}, "config": {"retrieve_k": 12}, "rows": rows},
                          rescore.Path("src.json"), scored, changes)
    assert report["rescored"]["from"] == "src.json"
    assert "recall@12" in report["summary"]      # ks taken from the stored retrieve_k


# --- the refusals ---

def test_refuses_a_report_with_no_stored_retrieval(tmp_path, capsys):
    """The precondition that stops a pre-`candidate_pages` report scoring as all zeros."""
    old = {"summary": {}, "config": {}, "rows": [{"id": "q1", "tags": [], "found": True,
                                                  "gold_rank": 1, "answer": ""}]}
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    code = rescore.main([str(path), "--output", str(tmp_path / "out.json")])
    assert code == 2
    assert "candidate_pages" in capsys.readouterr().out
    assert not (tmp_path / "out.json").exists()


def test_refuses_a_degraded_report(tmp_path, capsys):
    path = tmp_path / "deg.json"
    path.write_text(json.dumps({"degraded_run": True, "summary": {}, "rows": []}))
    assert rescore.main([str(path), "--output", str(tmp_path / "out.json")]) == 2
    assert "degraded_run" in capsys.readouterr().out


@pytest.mark.parametrize("config,expected", [
    ({"retrieve_k": 12}, (1, 3, 12)),
    ({"retrieve_k": 3}, (1, 3)),
    ({}, (1, 3, 10)),
])
def test_recall_ks_follow_the_source_report(config, expected):
    """A rescored report must carry the same metric *names* or diff_reports can't join it."""
    assert rescore._ks({"config": config, "summary": {}}) == expected


def test_recall_ks_fall_back_to_the_stored_summary_keys():
    assert rescore._ks({"summary": {"recall@1": 1.0, "recall@12": 1.0}}) == (1, 12)
