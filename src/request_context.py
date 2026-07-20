"""Per-request observability state, held in contextvars.

One place for the cross-cutting state a single query threads through the pipeline:
a `request_id` - bound once in `run_query` and stamped onto every log line by
`logging_setup._RequestIdFilter` - and a running token/cost accumulator fed by
`gemini_client._log_usage` and summarized when the query finishes.

Contextvars give per-thread / per-task isolation, so concurrent queries in a future
server each get their own id and totals with no shared mutable global. This module
imports nothing from the app so `logging_setup` can import it without a cycle.
"""

import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass

# "-" (not None) so lines emitted outside a query - ingest, startup - still carry a
# stable `request_id` field rather than a null, keeping the JSON log schema constant.
_request_id: ContextVar[str] = ContextVar("request_id", default="-")
# None outside a request; a fresh accumulator dict while one is active.
_usage: ContextVar[dict | None] = ContextVar("usage", default=None)
# Per-stage (graph node) accounting. `_stages` is the ordered list of stage buckets
# for the active request; `_current_stage` is the bucket Gemini usage is folded into
# right now - set by `enter_stage`, cleared by `exit_stage`. Both are None outside a
# request, so stage tracking is a no-op for direct/CLI calls that never begin one.
_stages: ContextVar[list | None] = ContextVar("stages", default=None)
_current_stage: ContextVar[dict | None] = ContextVar("current_stage", default=None)


@dataclass
class _Scope:
    """Opaque handle from `begin_request`; carries the reset tokens for every var."""

    request_token: Token
    usage_token: Token
    stages_token: Token
    current_stage_token: Token


def _fresh_usage() -> dict:
    return {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "est_cost_usd": 0.0,
        "gemini_calls": 0,
    }


def _fresh_stage(name: str) -> dict:
    """A per-stage bucket: the node's name + latency, plus the same usage keys as the
    request accumulator so totals and stages share one shape."""
    return {"node": name, "latency_ms": 0.0, **_fresh_usage()}


def current_request_id() -> str:
    """The request_id bound to the current context, or '-' outside a query."""
    return _request_id.get()


def begin_request() -> _Scope:
    """Start a request scope: bind a fresh uuid request_id and a zeroed accumulator.

    Returns a scope handle to hand back to `end_request` for a clean reset.
    """
    request_token = _request_id.set(uuid.uuid4().hex)
    usage_token = _usage.set(_fresh_usage())
    stages_token = _stages.set([])
    current_stage_token = _current_stage.set(None)
    return _Scope(request_token, usage_token, stages_token, current_stage_token)


def end_request(scope: _Scope) -> None:
    """Reset every contextvar to what it was before `begin_request`."""
    _request_id.reset(scope.request_token)
    _usage.reset(scope.usage_token)
    _stages.reset(scope.stages_token)
    _current_stage.reset(scope.current_stage_token)


def record_usage(*, prompt: int, output: int, total: int, cost: float) -> None:
    """Fold one Gemini call's tokens/cost into the request accumulator and, when a
    graph stage is executing, that stage's bucket too.

    A no-op when no request is active (a stray call outside `run_query`), so it never
    raises. Both accumulators share the `_fresh_usage` key shape, so one loop folds
    into whichever are live.
    """
    for acc in (_usage.get(), _current_stage.get()):
        if acc is None:
            continue
        acc["prompt_tokens"] += prompt
        acc["output_tokens"] += output
        acc["total_tokens"] += total
        acc["est_cost_usd"] = round(acc["est_cost_usd"] + cost, 6)
        acc["gemini_calls"] += 1


def usage_totals() -> dict:
    """A copy of the active request's accumulated totals (empty dict if none)."""
    acc = _usage.get()
    return dict(acc) if acc is not None else {}


def enter_stage(name: str) -> None:
    """Open a stage bucket for graph node `name` and make it the usage target.

    Called by `graph._timed` on node entry; `record_usage` then attributes each Gemini
    call to this stage until `exit_stage`. A no-op outside a request (`_stages` is None),
    so directly-called nodes in tests never accumulate stages.
    """
    stages = _stages.get()
    if stages is None:
        return
    bucket = _fresh_stage(name)
    stages.append(bucket)
    _current_stage.set(bucket)


def exit_stage(name: str, latency_ms: float) -> None:
    """Close the current stage, stamping its wall-clock latency. No-op outside a request."""
    stage = _current_stage.get()
    if stage is not None and stage["node"] == name:
        stage["latency_ms"] = latency_ms
    _current_stage.set(None)


def stage_breakdown() -> list[dict]:
    """A copy of the active request's ordered per-stage buckets (empty list if none)."""
    stages = _stages.get()
    return [dict(s) for s in stages] if stages is not None else []
