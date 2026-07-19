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


@dataclass
class _Scope:
    """Opaque handle from `begin_request`; carries the reset tokens for both vars."""

    request_token: Token
    usage_token: Token


def _fresh_usage() -> dict:
    return {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "est_cost_usd": 0.0,
        "gemini_calls": 0,
    }


def current_request_id() -> str:
    """The request_id bound to the current context, or '-' outside a query."""
    return _request_id.get()


def begin_request() -> _Scope:
    """Start a request scope: bind a fresh uuid request_id and a zeroed accumulator.

    Returns a scope handle to hand back to `end_request` for a clean reset.
    """
    request_token = _request_id.set(uuid.uuid4().hex)
    usage_token = _usage.set(_fresh_usage())
    return _Scope(request_token, usage_token)


def end_request(scope: _Scope) -> None:
    """Reset both contextvars to what they were before `begin_request`."""
    _request_id.reset(scope.request_token)
    _usage.reset(scope.usage_token)


def record_usage(*, prompt: int, output: int, total: int, cost: float) -> None:
    """Fold one Gemini call's tokens/cost into the active request's accumulator.

    A no-op when no request is active (a stray call outside `run_query`), so it never
    raises.
    """
    acc = _usage.get()
    if acc is None:
        return
    acc["prompt_tokens"] += prompt
    acc["output_tokens"] += output
    acc["total_tokens"] += total
    acc["est_cost_usd"] = round(acc["est_cost_usd"] + cost, 6)
    acc["gemini_calls"] += 1


def usage_totals() -> dict:
    """A copy of the active request's accumulated totals (empty dict if none)."""
    acc = _usage.get()
    return dict(acc) if acc is not None else {}
