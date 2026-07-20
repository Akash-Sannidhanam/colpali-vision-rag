"""Tests for the eval harness's pure scoring logic (eval.scoring).

Plain dicts in, plain values out - no models, network, files, or `src.` imports,
matching the repo's stub-the-choke-point testing conventions. The run_eval CLI
orchestration (Qdrant, Gemini judge) is exercised separately.
"""

import pytest

from eval.scoring import (
    aggregate,
    citation_correct,
    format_table,
    gold_rank,
    load_dataset,
    substring_match,
)


def _hit(pdf: str, page: int) -> dict:
    return {"pdf": pdf, "page_number": page, "image_path": f"{pdf}_{page}.png", "score": 0.5}


GOLD_A2 = [{"pdf": "a.pdf", "page": 2}]


# ---------------------------------------------------------------- gold_rank

def test_gold_rank_is_1_based_first_match():
    hits = [_hit("a.pdf", 1), _hit("a.pdf", 2), _hit("b.pdf", 1)]
    assert gold_rank(hits, GOLD_A2) == 2


def test_gold_rank_requires_pdf_and_page_to_match():
    # Page 2 exists but in the wrong pdf; a.pdf is present at the wrong page.
    hits = [_hit("b.pdf", 2), _hit("a.pdf", 1)]
    assert gold_rank(hits, GOLD_A2) is None


def test_gold_rank_multiple_gold_pages_returns_earliest_hit():
    gold = [{"pdf": "a.pdf", "page": 5}, {"pdf": "a.pdf", "page": 1}]
    hits = [_hit("a.pdf", 1), _hit("a.pdf", 5)]
    assert gold_rank(hits, gold) == 1


def test_gold_rank_empty_hits_is_none():
    assert gold_rank([], GOLD_A2) is None


# ---------------------------------------------------------- citation_correct

def _citation(found=True, source_page=1) -> dict:
    return {"answer": "x", "found": found, "source_page": source_page, "box": [0, 0, 1, 1]}


def test_citation_correct_resolves_source_page_against_reranked_list():
    reranked = [_hit("b.pdf", 9), _hit("a.pdf", 2)]
    assert citation_correct(_citation(source_page=2), reranked, GOLD_A2) is True


def test_citation_wrong_page_is_incorrect():
    reranked = [_hit("b.pdf", 9), _hit("a.pdf", 2)]
    assert citation_correct(_citation(source_page=1), reranked, GOLD_A2) is False


def test_citation_not_found_is_incorrect():
    reranked = [_hit("a.pdf", 2)]
    assert citation_correct(_citation(found=False, source_page=0), reranked, GOLD_A2) is False


def test_citation_source_page_out_of_range_is_incorrect():
    reranked = [_hit("a.pdf", 2)]
    assert citation_correct(_citation(source_page=2), reranked, GOLD_A2) is False
    assert citation_correct(_citation(source_page=0), reranked, GOLD_A2) is False


def test_citation_none_is_incorrect():
    assert citation_correct(None, [_hit("a.pdf", 2)], GOLD_A2) is False


# ---------------------------------------------------------- substring_match

def test_substring_match_is_case_insensitive():
    assert substring_match("Total Revenue: $180M", ["revenue"]) is True


def test_substring_match_any_of():
    assert substring_match("about 180 units", ["units sold", "180"]) is True


def test_substring_match_miss():
    assert substring_match("no idea", ["180"]) is False


def test_substring_match_without_expected_is_not_applicable():
    assert substring_match("anything", None) is None
    assert substring_match("anything", []) is None


# ---------------------------------------------------------------- load_dataset

VALID_ROW = (
    '{"id": "q1", "question": "What was Q4 revenue?",'
    ' "gold": [{"pdf": "a.pdf", "page": 2}], "answer_contains": ["180"], "tags": ["chart"]}'
)


def test_load_dataset_parses_and_fills_defaults():
    rows = load_dataset([
        VALID_ROW,
        '{"id": "q2", "question": "Who?", "gold": [{"pdf": "b.pdf", "page": 1}]}',
    ])
    assert [r["id"] for r in rows] == ["q1", "q2"]
    assert rows[0]["answer_contains"] == ["180"]
    assert rows[1]["answer_contains"] is None
    assert rows[1]["tags"] == []


def test_load_dataset_skips_blank_lines():
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
    with pytest.raises(ValueError, match="line 2"):
        load_dataset([VALID_ROW, bad_line])


def test_load_dataset_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="q1"):
        load_dataset([VALID_ROW, VALID_ROW])


# ------------------------------------------------------------------ aggregate

def test_aggregate_recall_at_ks_from_gold_rank():
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
    rows = [{"id": "a", "tags": [], "gold_rank": 1}]
    summary = aggregate(rows, ks=(1,))
    assert summary["rerank_recall"] is None
    assert summary["citation_accuracy"] is None
    assert summary["substring_accuracy"] is None
    assert summary["judge_accuracy"] is None


def test_aggregate_boolean_and_judge_metrics():
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


def test_aggregate_average_latency():
    rows = [
        {"id": "a", "tags": [], "gold_rank": 1, "latency_ms": 100.0},
        {"id": "b", "tags": [], "gold_rank": 1, "latency_ms": 300.0},
    ]
    assert aggregate(rows, ks=(1,))["avg_latency_ms"] == 200.0


# --------------------------------------------------------------- format_table

def test_format_table_smoke():
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
