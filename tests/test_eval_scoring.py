"""Tests for the eval harness's pure scoring logic (eval.scoring).

Plain dicts in, plain values out - no models, network, files, or `src.` imports,
matching the repo's stub-the-choke-point testing conventions. The run_eval CLI
orchestration (Qdrant, Gemini judge) is exercised separately.
"""

import pytest

from eval.scoring import (
    abstention_correct,
    aggregate,
    citation_correct,
    format_table,
    gold_doc_coverage,
    gold_rank,
    load_dataset,
    substring_match,
)


def _hit(pdf: str, page: int) -> dict:
    """One retrieval hit shaped like vector_store.search output."""
    return {"pdf": pdf, "page_number": page, "image_path": f"{pdf}_{page}.png", "score": 0.5}


GOLD_A2 = [{"pdf": "a.pdf", "page": 2}]


# ---------------------------------------------------------------- gold_rank

def test_gold_rank_is_1_based_first_match():
    """Rank is 1-based and reports the first gold hit's position."""
    hits = [_hit("a.pdf", 1), _hit("a.pdf", 2), _hit("b.pdf", 1)]
    assert gold_rank(hits, GOLD_A2) == 2


def test_gold_rank_requires_pdf_and_page_to_match():
    """A right page in the wrong document is not a hit."""
    # Page 2 exists but in the wrong pdf; a.pdf is present at the wrong page.
    hits = [_hit("b.pdf", 2), _hit("a.pdf", 1)]
    assert gold_rank(hits, GOLD_A2) is None


def test_gold_rank_multiple_gold_pages_returns_earliest_hit():
    """With several gold pages, the earliest-ranked hit wins."""
    gold = [{"pdf": "a.pdf", "page": 5}, {"pdf": "a.pdf", "page": 1}]
    hits = [_hit("a.pdf", 1), _hit("a.pdf", 5)]
    assert gold_rank(hits, gold) == 1


def test_gold_rank_empty_hits_is_none():
    """No hits means no rank, not rank 0."""
    assert gold_rank([], GOLD_A2) is None


# ---------------------------------------------------------- citation_correct

def _citation(found=True, source_page=1) -> dict:
    """A citation dict with the given found flag and 1-based source_page."""
    return {"answer": "x", "found": found, "source_page": source_page, "box": [0, 0, 1, 1]}


def test_citation_correct_resolves_source_page_against_reranked_list():
    """source_page indexes the reranked list, which is what the answer step saw."""
    reranked = [_hit("b.pdf", 9), _hit("a.pdf", 2)]
    assert citation_correct(_citation(source_page=2), reranked, GOLD_A2) is True


def test_citation_wrong_page_is_incorrect():
    """Citing a non-gold page is scored incorrect."""
    reranked = [_hit("b.pdf", 9), _hit("a.pdf", 2)]
    assert citation_correct(_citation(source_page=1), reranked, GOLD_A2) is False


def test_citation_not_found_is_incorrect():
    """A not-found citation never counts as a correct citation."""
    reranked = [_hit("a.pdf", 2)]
    assert citation_correct(_citation(found=False, source_page=0), reranked, GOLD_A2) is False


def test_citation_source_page_out_of_range_is_incorrect():
    """An out-of-range index is incorrect rather than an IndexError."""
    reranked = [_hit("a.pdf", 2)]
    assert citation_correct(_citation(source_page=2), reranked, GOLD_A2) is False
    assert citation_correct(_citation(source_page=0), reranked, GOLD_A2) is False


def test_citation_none_is_incorrect():
    """A missing citation scores incorrect instead of raising."""
    assert citation_correct(None, [_hit("a.pdf", 2)], GOLD_A2) is False


# ---------------------------------------------------------- substring_match

# --------------------------------------------------------- abstention_correct

def test_abstention_correct_only_when_the_system_declined():
    """Declining scores correct; answering an unanswerable question does not."""
    assert abstention_correct({"found": False, "source_page": 0}) is True
    assert abstention_correct({"found": True, "source_page": 1}) is False


def test_abstention_correct_treats_a_missing_citation_as_declining():
    """The degraded not-found shape (no citation at all) still counts as abstaining."""
    assert abstention_correct(None) is True
    assert abstention_correct({}) is True


# --------------------------------------------------------- gold_doc_coverage

def test_gold_doc_coverage_is_not_applicable_within_one_document():
    """Single-document gold has no cross-document coverage to report."""
    reranked = [_hit("a.pdf", 2)]
    assert gold_doc_coverage(reranked, GOLD_A2) is None
    assert gold_doc_coverage(reranked, [{"pdf": "a.pdf", "page": 2},
                                        {"pdf": "a.pdf", "page": 5}]) is None


CROSS_GOLD = [{"pdf": "a.pdf", "page": 2}, {"pdf": "b.pdf", "page": 7}]


def test_gold_doc_coverage_full_when_both_documents_are_reached():
    """Both gold documents represented by a gold page scores 1.0."""
    assert gold_doc_coverage([_hit("a.pdf", 2), _hit("b.pdf", 7)], CROSS_GOLD) == 1.0


def test_gold_doc_coverage_half_when_only_one_document_is_reached():
    """Spending both rerank slots inside one document scores 0.5."""
    assert gold_doc_coverage([_hit("a.pdf", 2), _hit("a.pdf", 3)], CROSS_GOLD) == 0.5


def test_gold_doc_coverage_needs_the_gold_page_not_just_the_document():
    """The right document at the wrong page does not count as covered."""
    assert gold_doc_coverage([_hit("a.pdf", 2), _hit("b.pdf", 1)], CROSS_GOLD) == 0.5


def test_substring_match_is_case_insensitive():
    """Expected substrings match regardless of case."""
    assert substring_match("Total Revenue: $180M", ["revenue"]) is True


def test_substring_match_any_of():
    """Any one of the expected substrings is enough to match."""
    assert substring_match("about 180 units", ["units sold", "180"]) is True


def test_substring_match_miss():
    """No expected substring present scores False."""
    assert substring_match("no idea", ["180"]) is False


def test_substring_match_ignores_thousands_separators():
    """A digit-grouping comma is typography, not a different number.

    The live eval scored "37,000" against the label "37000" as a miss, which measured
    formatting rather than correctness.
    """
    assert substring_match("about 37,000 tokens", ["37000"]) is True
    assert substring_match("about 37000 tokens", ["37,000"]) is True
    assert substring_match("$1,234,567 in revenue", ["1234567"]) is True
    # A comma that isn't grouping thousands is left alone, so 37,0 stays distinct.
    assert substring_match("37,0 units", ["370"]) is False


def test_substring_match_without_expected_is_not_applicable():
    """No expectation yields None (N/A), which aggregation excludes from denominators."""
    assert substring_match("anything", None) is None
    assert substring_match("anything", []) is None


# ---------------------------------------------------------------- load_dataset

VALID_ROW = (
    '{"id": "q1", "question": "What was Q4 revenue?",'
    ' "gold": [{"pdf": "a.pdf", "page": 2}], "answer_contains": ["180"], "tags": ["chart"]}'
)


def test_load_dataset_parses_and_fills_defaults():
    """Rows parse and optional fields take their documented defaults."""
    rows = load_dataset([
        VALID_ROW,
        '{"id": "q2", "question": "Who?", "gold": [{"pdf": "b.pdf", "page": 1}]}',
    ])
    assert [r["id"] for r in rows] == ["q1", "q2"]
    assert rows[0]["answer_contains"] == ["180"]
    assert rows[1]["answer_contains"] is None
    assert rows[1]["tags"] == []


def test_load_dataset_skips_blank_lines():
    """Blank and whitespace-only lines are ignored, not treated as bad rows."""
    rows = load_dataset(["", VALID_ROW, "   "])
    assert len(rows) == 1


@pytest.mark.parametrize(
    "bad_line",
    [
        "not json",
        '{"question": "no id", "gold": [{"pdf": "a.pdf", "page": 1}]}',
        '{"id": "q9", "gold": [{"pdf": "a.pdf", "page": 1}]}',
        '{"id": "q9", "question": "no gold"}',
        '{"id": "q9", "question": "empty gold", "gold": []}',
        '{"id": "q9", "question": "bad page", "gold": [{"pdf": "a.pdf", "page": 0}]}',
        '{"id": "q9", "question": "gold not list", "gold": {"pdf": "a.pdf", "page": 1}}',
    ],
)
def test_load_dataset_rejects_invalid_rows_naming_the_line(bad_line):
    """A malformed row raises an error naming the offending line number."""
    with pytest.raises(ValueError, match="line 2"):
        load_dataset([VALID_ROW, bad_line])


def test_load_dataset_rejects_duplicate_ids():
    """Duplicate question ids are rejected, naming the repeated id."""
    with pytest.raises(ValueError, match="q1"):
        load_dataset([VALID_ROW, VALID_ROW])


def test_load_dataset_defaults_unanswerable_to_false():
    """An ordinary row is answerable unless it says otherwise."""
    assert load_dataset([VALID_ROW])[0]["unanswerable"] is False


def test_load_dataset_accepts_unanswerable_rows_with_or_without_empty_gold():
    """`unanswerable` inverts the gold rule: absent and explicitly-empty both parse."""
    rows = load_dataset([
        '{"id": "n1", "question": "Q1 2026 revenue?", "unanswerable": true, "gold": []}',
        '{"id": "n2", "question": "En-Ru BLEU?", "unanswerable": true}',
    ])
    assert [r["unanswerable"] for r in rows] == [True, True]
    assert [r["gold"] for r in rows] == [[], []]


@pytest.mark.parametrize(
    "bad_line",
    [
        # An unanswerable question cannot also name where the answer is...
        '{"id": "n9", "question": "x", "unanswerable": true,'
        ' "gold": [{"pdf": "a.pdf", "page": 1}]}',
        # ...nor expect any substring in the answer.
        '{"id": "n9", "question": "x", "unanswerable": true, "answer_contains": ["180"]}',
        '{"id": "n9", "question": "x", "unanswerable": "yes"}',
    ],
)
def test_load_dataset_rejects_contradictory_unanswerable_rows(bad_line):
    """A row that claims to be unanswerable while labelling an answer is a mistake."""
    with pytest.raises(ValueError, match="line 2"):
        load_dataset([VALID_ROW, bad_line])


# ------------------------------------------------------------------ aggregate

def test_aggregate_recall_at_ks_from_gold_rank():
    """recall@k is derived from gold_rank across the requested cutoffs."""
    rows = [
        {"id": "a", "tags": [], "gold_rank": 1},
        {"id": "b", "tags": [], "gold_rank": 4},
        {"id": "c", "tags": [], "gold_rank": None},
        {"id": "d", "tags": [], "gold_rank": 2},
    ]
    summary = aggregate(rows, ks=(1, 3, 10))
    assert summary["n"] == 4
    assert summary["recall@1"] == 0.25
    assert summary["recall@3"] == 0.5
    assert summary["recall@10"] == 0.75


def test_aggregate_excludes_not_applicable_rows_from_denominators():
    """N/A and absent values are excluded, so rates are over applicable rows only."""
    rows = [
        {"id": "a", "tags": [], "gold_rank": 1, "substring_match": True},
        {"id": "b", "tags": [], "gold_rank": 1, "substring_match": None},
        {"id": "c", "tags": [], "gold_rank": 1, "substring_match": False},
        {"id": "d", "tags": [], "gold_rank": 1},
    ]
    summary = aggregate(rows, ks=(1,))
    # 2 applicable rows (True, False) -> 0.5; None / absent rows don't count.
    assert summary["substring_accuracy"] == 0.5


def test_aggregate_metric_with_no_applicable_rows_is_none():
    """A metric with nothing to score reports None rather than a misleading 0.0."""
    rows = [{"id": "a", "tags": [], "gold_rank": 1}]
    summary = aggregate(rows, ks=(1,))
    assert summary["rerank_recall"] is None
    assert summary["citation_accuracy"] is None
    assert summary["substring_accuracy"] is None
    assert summary["judge_accuracy"] is None


def test_aggregate_boolean_and_judge_metrics():
    """Boolean metrics average as rates and judge scores average numerically."""
    rows = [
        {"id": "a", "tags": [], "gold_rank": 1, "rerank_hit": True,
         "citation_correct": True, "judge": {"correct": True, "score": 5}},
        {"id": "b", "tags": [], "gold_rank": 1, "rerank_hit": False,
         "citation_correct": True, "judge": {"correct": False, "score": 1}},
    ]
    summary = aggregate(rows, ks=(1,))
    assert summary["rerank_recall"] == 0.5
    assert summary["citation_accuracy"] == 1.0
    assert summary["judge_accuracy"] == 0.5
    assert summary["judge_score_avg"] == 3.0


def test_aggregate_per_tag_slices():
    """Per-tag slices cover only tagged rows while the overall rate still counts them all."""
    rows = [
        {"id": "a", "tags": ["chart"], "gold_rank": 1},
        {"id": "b", "tags": ["chart", "table"], "gold_rank": None},
        {"id": "c", "tags": [], "gold_rank": 1},
    ]
    summary = aggregate(rows, ks=(1,))
    assert summary["per_tag"]["chart"]["n"] == 2
    assert summary["per_tag"]["chart"]["recall@1"] == 0.5
    assert summary["per_tag"]["table"]["n"] == 1
    assert summary["per_tag"]["table"]["recall@1"] == 0.0
    # Untagged rows appear in no slice; overall recall still counts them.
    assert set(summary["per_tag"]) == {"chart", "table"}
    assert summary["recall@1"] == pytest.approx(2 / 3, abs=1e-4)


def test_aggregate_keeps_unanswerable_rows_out_of_recall_and_citation():
    """Negative rows score abstention only; they never dilute the retrieval metrics.

    This is the whole point of the row-shape split in run_eval.run_full: an
    unanswerable question has no gold page, so counting it as a recall miss would
    make the pipeline look worse the more honest it got.
    """
    rows = [
        {"id": "a", "tags": [], "gold_rank": 1, "citation_correct": True},
        {"id": "n1", "tags": ["unanswerable"], "abstention_correct": True},
        {"id": "n2", "tags": ["unanswerable"], "abstention_correct": False},
    ]
    summary = aggregate(rows, ks=(1,))
    assert summary["n"] == 3                    # every row is still counted...
    assert summary["recall@1"] == 1.0           # ...over the 1 row that has a gold page
    assert summary["citation_accuracy"] == 1.0
    assert summary["abstention_accuracy"] == 0.5
    assert summary["per_tag"]["unanswerable"]["recall@1"] is None


def test_aggregate_gold_coverage_averages_only_cross_document_rows():
    """gold_coverage_avg spans the rows where coverage was applicable at all."""
    rows = [
        {"id": "a", "tags": [], "gold_rank": 1, "gold_doc_coverage": 1.0},
        {"id": "b", "tags": [], "gold_rank": 1, "gold_doc_coverage": 0.5},
        {"id": "c", "tags": [], "gold_rank": 1, "gold_doc_coverage": None},
    ]
    assert aggregate(rows, ks=(1,))["gold_coverage_avg"] == 0.75


def test_aggregate_confidence_separation_splits_on_citation_correctness():
    """Calibration reports the gap between confidence on right vs wrong citations."""
    rows = [
        {"id": "a", "tags": [], "citation_correct": True, "retrieval_confidence": 0.9,
         "self_confidence": "high"},
        {"id": "b", "tags": [], "citation_correct": True, "retrieval_confidence": 0.7,
         "self_confidence": "high"},
        {"id": "c", "tags": [], "citation_correct": False, "retrieval_confidence": 0.3,
         "self_confidence": "low"},
    ]
    summary = aggregate(rows, ks=(1,))
    assert summary["retrieval_conf_correct_avg"] == 0.8
    assert summary["retrieval_conf_wrong_avg"] == 0.3
    assert summary["confidence_separation"] == 0.5
    assert summary["self_conf_high_acc"] == 1.0
    assert summary["self_conf_low_acc"] == 0.0
    assert summary["self_conf_medium_acc"] is None


def test_aggregate_confidence_separation_is_none_without_both_sides():
    """A slice with no wrong citations has no separation to report - the saturated case."""
    rows = [
        {"id": "a", "tags": [], "citation_correct": True, "retrieval_confidence": 0.9},
        {"id": "b", "tags": [], "citation_correct": True, "retrieval_confidence": 0.7},
    ]
    summary = aggregate(rows, ks=(1,))
    assert summary["retrieval_conf_correct_avg"] == 0.8
    assert summary["retrieval_conf_wrong_avg"] is None
    assert summary["confidence_separation"] is None


def test_aggregate_average_latency():
    """avg_latency_ms is the mean over rows that reported a latency."""
    rows = [
        {"id": "a", "tags": [], "gold_rank": 1, "latency_ms": 100.0},
        {"id": "b", "tags": [], "gold_rank": 1, "latency_ms": 300.0},
    ]
    assert aggregate(rows, ks=(1,))["avg_latency_ms"] == 200.0


# --------------------------------------------------------------- format_table

def test_format_table_smoke():
    """The rendered table includes row ids, the recall columns, and dashes for N/A cells."""
    rows = [
        {"id": "sales-q4", "tags": ["chart"], "gold_rank": 1, "rerank_hit": True,
         "citation_correct": True, "substring_match": None, "latency_ms": 1234.5},
        {"id": "miss", "tags": [], "gold_rank": None},
    ]
    summary = aggregate(rows, ks=(1, 3, 10))
    table = format_table(rows, summary)
    assert "sales-q4" in table
    assert "miss" in table
    assert "recall@10" in table
    assert "-" in table  # N/A cells render as a dash
