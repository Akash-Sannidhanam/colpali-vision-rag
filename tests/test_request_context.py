"""Tests for src.request_context - the per-request contextvar state.

Pure logic: no models, API key, or network. Verifies request_id binding/reset and
the token/cost accumulator's math plus its no-op-outside-a-request behavior.
"""

from src import request_context


def test_begin_request_binds_id_and_end_restores_default():
    assert request_context.current_request_id() == "-"
    scope = request_context.begin_request()
    rid = request_context.current_request_id()
    assert rid != "-" and len(rid) == 32          # uuid4().hex
    request_context.end_request(scope)
    assert request_context.current_request_id() == "-"


def test_record_usage_accumulates_and_totals_sum():
    scope = request_context.begin_request()
    try:
        request_context.record_usage(prompt=1000, output=500, total=1500, cost=0.001)
        request_context.record_usage(prompt=200, output=100, total=300, cost=0.0005)
        totals = request_context.usage_totals()
    finally:
        request_context.end_request(scope)

    assert totals["prompt_tokens"] == 1200
    assert totals["output_tokens"] == 600
    assert totals["total_tokens"] == 1800
    assert totals["est_cost_usd"] == round(0.001 + 0.0005, 6)
    assert totals["gemini_calls"] == 2


def test_fresh_request_resets_totals():
    first = request_context.begin_request()
    request_context.record_usage(prompt=10, output=5, total=15, cost=0.0)
    request_context.end_request(first)

    second = request_context.begin_request()
    try:
        assert request_context.usage_totals()["total_tokens"] == 0
    finally:
        request_context.end_request(second)


def test_record_usage_outside_request_is_noop():
    # No active request -> no accumulator -> must not raise, totals stay empty.
    request_context.record_usage(prompt=1, output=1, total=2, cost=0.1)
    assert request_context.usage_totals() == {}


def test_stage_breakdown_buckets_usage_by_current_stage():
    scope = request_context.begin_request()
    try:
        request_context.enter_stage("rerank")
        request_context.record_usage(prompt=100, output=50, total=150, cost=0.001)
        request_context.exit_stage("rerank", 12.0)
        request_context.enter_stage("answer")
        request_context.record_usage(prompt=200, output=100, total=300, cost=0.002)
        request_context.exit_stage("answer", 34.0)
        stages = request_context.stage_breakdown()
        totals = request_context.usage_totals()
    finally:
        request_context.end_request(scope)

    assert [s["node"] for s in stages] == ["rerank", "answer"]
    rerank, answer = stages
    assert rerank["latency_ms"] == 12.0
    assert rerank["total_tokens"] == 150 and rerank["gemini_calls"] == 1
    assert answer["latency_ms"] == 34.0 and answer["total_tokens"] == 300
    # each Gemini call folds into BOTH its stage and the request aggregate
    assert totals["total_tokens"] == 450 and totals["gemini_calls"] == 2


def test_usage_outside_a_stage_hits_the_aggregate_only():
    scope = request_context.begin_request()
    try:
        request_context.record_usage(prompt=10, output=5, total=15, cost=0.0)
        stages = request_context.stage_breakdown()
        totals = request_context.usage_totals()
    finally:
        request_context.end_request(scope)

    assert stages == []                               # no enter_stage -> no bucket
    assert totals["total_tokens"] == 15               # but the total still counts it


def test_stage_helpers_are_noop_outside_a_request():
    request_context.enter_stage("rerank")             # must not raise with no scope
    request_context.exit_stage("rerank", 1.0)
    assert request_context.stage_breakdown() == []
