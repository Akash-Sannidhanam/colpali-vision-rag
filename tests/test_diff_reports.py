"""Tests for eval.diff_reports - the paired before/after report comparison.

Pure logic over hand-built report dicts: no files, no pipeline, no network. The
comparison is what decides whether a knob change ships, so its arithmetic and - more
importantly - its refusal to compare runs that aren't comparable are worth pinning.
"""

import pytest

from eval.diff_reports import (
    METRIC_DIRECTION,
    compare,
    config_changes,
    format_diff,
    row_metric,
)


def _report(rows, summary=None, config=None):
    """A minimal report in the shape run_eval writes."""
    return {
        "config": {"rerank_k": 2, "dataset": "eval/dataset.jsonl", **(config or {})},
        "summary": {"gold_coverage_avg": 0.75, **(summary or {})},
        "rows": rows,
    }


def _row(row_id, **fields):
    """One scored row."""
    return {"id": row_id, "tags": ["cross-doc"], **fields}


# --- row_metric: reading one value out of a row ---

def test_row_metric_reads_a_plain_field():
    assert row_metric(_row("q1", gold_doc_coverage=0.5), "gold_doc_coverage") == 0.5


def test_row_metric_reads_a_dotted_path():
    """`judge.correct` lives one level down; the judge block is the only nested one."""
    assert row_metric(_row("q1", judge={"correct": True, "score": 5}), "judge.correct") is True


def test_row_metric_of_a_missing_field_is_none():
    """An N/A metric reads as None rather than raising - most rows are N/A for most metrics."""
    assert row_metric(_row("q1"), "gold_doc_coverage") is None


def test_row_metric_survives_a_null_judge():
    """A judge N/A is None, not a crash on attribute lookup into None."""
    assert row_metric(_row("q1", judge=None), "judge.correct") is None


# --- compare: the paired join ---

def test_compare_pairs_rows_by_id_and_reports_flips():
    """The core: same question, both sides, changed value."""
    before = _report([_row("x1", gold_doc_coverage=0.5), _row("x2", gold_doc_coverage=1.0)])
    after = _report([_row("x1", gold_doc_coverage=1.0), _row("x2", gold_doc_coverage=1.0)])

    out = compare(before, after, "gold_doc_coverage")

    assert [(f["id"], f["before"], f["after"]) for f in out["flips"]] == [("x1", 0.5, 1.0)]
    assert out["improved"] == 1 and out["regressed"] == 0
    assert out["unchanged"] == 1


def test_compare_counts_a_regression_separately():
    """A net-zero change made of one gain and one loss is not the same as no change."""
    before = _report([_row("x1", gold_doc_coverage=0.5), _row("x2", gold_doc_coverage=1.0)])
    after = _report([_row("x1", gold_doc_coverage=1.0), _row("x2", gold_doc_coverage=0.5)])

    out = compare(before, after, "gold_doc_coverage")

    assert out["improved"] == 1 and out["regressed"] == 1
    assert out["before_avg"] == out["after_avg"] == 0.75   # the average hides it; the flips don't


def test_compare_respects_lower_is_better_for_gold_rank():
    """gold_rank improves by going *down*; a direction table keeps the sign honest."""
    assert METRIC_DIRECTION["gold_rank"] == -1
    before = _report([_row("x1", gold_rank=5)])
    after = _report([_row("x1", gold_rank=1)])

    out = compare(before, after, "gold_rank")

    assert out["improved"] == 1 and out["regressed"] == 0


def test_compare_scores_booleans_as_false_to_true_improvement():
    """citation_correct flipping True is an improvement, not an unorderable change."""
    before = _report([_row("x1", citation_correct=False)])
    after = _report([_row("x1", citation_correct=True)])

    out = compare(before, after, "citation_correct")

    assert out["improved"] == 1 and out["regressed"] == 0


def test_compare_ignores_rows_where_the_metric_is_na_on_both_sides():
    """Single-document rows are N/A for gold_doc_coverage and must not dilute the counts."""
    before = _report([_row("s1"), _row("x1", gold_doc_coverage=0.5)])
    after = _report([_row("s1"), _row("x1", gold_doc_coverage=1.0)])

    out = compare(before, after, "gold_doc_coverage")

    assert out["applicable"] == 1
    assert out["unchanged"] == 0


def test_compare_flags_a_row_that_became_na():
    """N/A on one side only is a shape change, and is surfaced rather than skipped."""
    before = _report([_row("x1", gold_doc_coverage=0.5)])
    after = _report([_row("x1")])

    out = compare(before, after, "gold_doc_coverage")

    assert [(f["id"], f["before"], f["after"]) for f in out["flips"]] == [("x1", 0.5, None)]
    # Neither an improvement nor a regression - there is no ordering against None.
    assert out["improved"] == 0 and out["regressed"] == 0


def test_compare_reports_ids_present_on_only_one_side():
    """Dataset drift must never masquerade as a metric change."""
    before = _report([_row("x1", gold_doc_coverage=0.5), _row("dropped", gold_doc_coverage=1.0)])
    after = _report([_row("x1", gold_doc_coverage=0.5), _row("added", gold_doc_coverage=1.0)])

    out = compare(before, after, "gold_doc_coverage")

    assert out["only_in_before"] == ["dropped"]
    assert out["only_in_after"] == ["added"]
    # Unpaired rows are excluded from the paired arithmetic entirely.
    assert out["applicable"] == 1


def test_compare_averages_only_the_paired_applicable_rows():
    """The averages must be over the same question set on both sides, or they aren't paired."""
    before = _report([_row("x1", gold_doc_coverage=0.5), _row("only_before", gold_doc_coverage=0.0)])
    after = _report([_row("x1", gold_doc_coverage=1.0)])

    out = compare(before, after, "gold_doc_coverage")

    assert out["before_avg"] == 0.5 and out["after_avg"] == 1.0   # 'only_before' excluded


def test_compare_rejects_a_metric_no_row_carries():
    """A typo'd metric fails loudly instead of reporting a serene zero flips."""
    before = _report([_row("x1", gold_doc_coverage=0.5)])
    after = _report([_row("x1", gold_doc_coverage=1.0)])

    with pytest.raises(ValueError, match="gold_coverage"):
        compare(before, after, "gold_coverage")       # the summary's name, not the row's


def test_compare_deltas_the_summary_metrics():
    """Summary movement still gets reported - it just isn't the evidence."""
    before = _report([_row("x1", gold_doc_coverage=0.5)], summary={"recall@1": 0.76})
    after = _report([_row("x1", gold_doc_coverage=1.0)], summary={"recall@1": 0.80})

    deltas = compare(before, after, "gold_doc_coverage")["summary_deltas"]

    assert deltas["recall@1"] == {"before": 0.76, "after": 0.8, "delta": 0.04}


def test_compare_skips_summary_metrics_that_are_na_on_either_side():
    """confidence_separation is None whenever the pipeline makes no wrong citation."""
    before = _report([_row("x1", gold_doc_coverage=0.5)], summary={"confidence_separation": None})
    after = _report([_row("x1", gold_doc_coverage=1.0)], summary={"confidence_separation": 0.03})

    assert "confidence_separation" not in compare(before, after, "gold_doc_coverage")["summary_deltas"]


# --- config_changes: what was actually different about the two runs ---

def test_config_changes_names_the_knob_that_moved():
    """The point of an arm comparison is that exactly one knob moved; show which."""
    before = _report([], config={"rerank_k": 2})
    after = _report([], config={"rerank_k": 3})

    assert config_changes(before, after) == {"rerank_k": (2, 3)}


def test_config_changes_is_empty_for_identical_configs():
    """Two runs of the same config - a variance check - report no knob change."""
    assert config_changes(_report([]), _report([])) == {}


def test_config_changes_catches_a_swapped_dataset():
    """Comparing runs over different datasets is the mistake most likely to be missed."""
    changes = config_changes(_report([], config={"dataset": "eval/a.jsonl"}), _report([]))
    assert "dataset" in changes


# --- degradation: a degraded run must never be silently compared ---

def test_compare_flags_a_degraded_side():
    """Half the point of the guard is not comparing against an outage after the fact."""
    before = _report([_row("x1", gold_doc_coverage=0.5)])
    after = _report([_row("x1", gold_doc_coverage=1.0)])
    after["degraded_run"] = True

    out = compare(before, after, "gold_doc_coverage")

    assert out["degraded_sides"] == ["after"]


def test_format_diff_warns_loudly_about_a_degraded_side():
    """The warning has to survive into the printed output, not just the dict."""
    before = _report([_row("x1", gold_doc_coverage=0.5)])
    after = _report([_row("x1", gold_doc_coverage=1.0)])
    before["degraded_run"] = True

    text = format_diff(compare(before, after, "gold_doc_coverage"), "gold_doc_coverage")

    assert "DEGRADED" in text


def test_format_diff_renders_flips_and_totals():
    """The human-facing output names the flipped question ids."""
    before = _report([_row("x1", gold_doc_coverage=0.5), _row("x2", gold_doc_coverage=0.5)])
    after = _report([_row("x1", gold_doc_coverage=1.0), _row("x2", gold_doc_coverage=0.5)])

    text = format_diff(compare(before, after, "gold_doc_coverage"), "gold_doc_coverage")

    assert "x1" in text
    assert "improved 1" in text and "regressed 0" in text
