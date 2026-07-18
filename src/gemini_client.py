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

from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

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


def _log_usage(response: Any, *, model: str, purpose: str) -> None:
    """Log token counts and an estimated cost for one call."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    prompt = getattr(usage, "prompt_token_count", 0) or 0
    output = getattr(usage, "candidates_token_count", 0) or 0
    total = getattr(usage, "total_token_count", 0) or 0
    in_rate, out_rate = _RATES.get(model, (0.0, 0.0))
    cost = prompt / 1e6 * in_rate + output / 1e6 * out_rate
    log.info(
        "gemini call",
        extra={
            "purpose": purpose,
            "model": model,
            "prompt_tokens": prompt,
            "output_tokens": output,
            "total_tokens": total,
            "est_cost_usd": round(cost, 6),
        },
    )


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

    response = _call()
    _log_usage(response, model=model, purpose=purpose)
    return response
