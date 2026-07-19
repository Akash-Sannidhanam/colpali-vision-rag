"""Single choke point for all Gemini traffic.

Every rerank/answer call goes through `generate()`, which gives the whole app,
in one place:
  - a cached client (callers no longer build one per request),
  - a request timeout,
  - retry with exponential backoff on transient errors (429 / 5xx / network),
  - token + estimated-cost logging per call.

`generate()` returns the raw SDK response, so callers keep using `.parsed` /
`.text` (and their own graceful-fallback handling) exactly as before.
"""

import logging
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src import request_context
from src.config import GEMINI_API_KEY, GEMINI_MAX_RETRIES, GEMINI_TIMEOUT_S
from src.logging_setup import get_logger

log = get_logger("gemini")

_client: genai.Client | None = None

# HTTP status codes worth retrying: rate limiting + transient server errors.
_RETRYABLE_CODES = {429, 500, 502, 503, 504}

# Rough USD per 1M tokens (input, output) for cost estimation only. Not billing
# accurate; update when pricing changes. Unknown models fall back to (0, 0).
_RATES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (0.30, 2.50),
}


def get_client() -> genai.Client:
    """Construct the Gemini client once (with a request timeout) and reuse it."""
    global _client
    if _client is None:
        # google-genai expresses the HTTP timeout in milliseconds.
        _client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=int(GEMINI_TIMEOUT_S * 1000)),
        )
    return _client


def _is_retryable(exc: BaseException) -> bool:
    """True for transient failures worth retrying, False for permanent ones.

    Retries rate-limit/5xx API errors and network timeouts/resets; does NOT
    retry auth (401/403), bad-request (400), or schema errors.
    """
    if isinstance(exc, genai_errors.APIError):
        return getattr(exc, "code", None) in _RETRYABLE_CODES
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name


def _log_usage(
    response: Any, *, model: str, purpose: str, latency_ms: float, attempts: int
) -> None:
    """Log one call's latency + attempt count, plus token usage when present.

    Always emits a `"gemini call"` line so latency/attempts stay observable even when
    the SDK returns no `usage_metadata` (`attempts > 1` is the retry signal). When
    usage is present, its token counts and estimated cost are added to the line and
    folded into the per-request totals via `record_usage`.
    """
    fields: dict[str, Any] = {
        "purpose": purpose,
        "model": model,
        "latency_ms": latency_ms,
        "attempts": attempts,
    }
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        prompt = getattr(usage, "prompt_token_count", 0) or 0
        output = getattr(usage, "candidates_token_count", 0) or 0
        total = getattr(usage, "total_token_count", 0) or 0
        in_rate, out_rate = _RATES.get(model, (0.0, 0.0))
        cost = prompt / 1e6 * in_rate + output / 1e6 * out_rate
        fields.update(
            prompt_tokens=prompt,
            output_tokens=output,
            total_tokens=total,
            est_cost_usd=round(cost, 6),
        )
        request_context.record_usage(
            prompt=prompt, output=output, total=total, cost=cost
        )
    log.info("gemini call", extra=fields)


def generate(*, model: str, contents: list, response_schema: Any, purpose: str) -> Any:
    """Structured-output generation with timeout + retry; logs token usage.

    Returns the SDK response object; callers read `.parsed` / `.text` as usual.
    `purpose` (e.g. "rerank" / "answer") tags the usage log line.
    """

    @retry(
        reraise=True,
        stop=stop_after_attempt(max(1, GEMINI_MAX_RETRIES)),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    def _call() -> Any:
        return get_client().models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

    start = time.perf_counter()
    response = _call()
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    attempts = _call.statistics.get("attempt_number", 1)
    _log_usage(
        response, model=model, purpose=purpose, latency_ms=latency_ms, attempts=attempts
    )
    return response
