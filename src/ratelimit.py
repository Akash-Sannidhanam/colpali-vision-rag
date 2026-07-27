"""Per-client-IP rate limiting for the serving layer.

In-process and dependency-free, which is the *correct* design here rather than a
compromise: the server is single-worker by construction (one GPU-resident ColQwen2
model serialized behind `server._gpu_lock`; the Dockerfile forbids `--workers >1`),
so one process sees every request and an in-memory counter is exact. A Redis-backed
limiter would add an operational dependency to count something only one process can
increment.

Two windows, because the two costs are different by orders of magnitude:

  * `RATE_LIMIT_PER_MINUTE` on the query/read endpoints - each `/query` is a GPU
    forward pass plus two paid Gemini calls.
  * `RATE_LIMIT_INGEST_PER_HOUR` on the ingest endpoints - each holds the model lock
    for minutes and re-renders a whole PDF.

Both are sliding windows (a deque of hit timestamps per client), not fixed buckets, so
a client can't burst 2x the limit across a bucket boundary. Rejected requests are not
recorded, so a client hammering a closed door still recovers exactly one window after
its last *accepted* request.
"""

import time
from collections import deque

from fastapi import HTTPException, Request

from src.config import (
    RATE_LIMIT_INGEST_PER_HOUR,
    RATE_LIMIT_PER_MINUTE,
    TRUST_PROXY_HEADERS,
)
from src.logging_setup import get_logger

log = get_logger("ratelimit")

# Indirection so tests can drive the window without sleeping. Monotonic, not wall
# clock, so an NTP step can't retroactively expire or extend a window.
_clock = time.monotonic

# Sweep idle clients out of a window once it is tracking more than this many. Bounds
# memory against a spray of one-request-each source addresses.
_SWEEP_AT = 1024


class _Window:
    """Sliding-window hit counter: one deque of timestamps per client key."""

    def __init__(self, window_s: float, unit: str):
        self.window_s = window_s
        self.unit = unit                                  # for the 429 message
        self._hits: dict[str, deque[float]] = {}

    def hit(self, client: str, limit: int, now: float) -> float | None:
        """Record one request for `client`.

        Returns None when it is within `limit`, or the seconds until the window frees
        a slot when it is not. A `limit` of 0 disables the window entirely. The hit is
        recorded only when it is accepted - see the module docstring.
        """
        if limit <= 0:
            return None

        cutoff = now - self.window_s
        hits = self._hits.setdefault(client, deque())
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= limit:
            return max(0.0, hits[0] + self.window_s - now)

        hits.append(now)
        if len(self._hits) > _SWEEP_AT:
            self._sweep(cutoff)
        return None

    def _sweep(self, cutoff: float) -> None:
        """Drop clients whose every hit has aged out of the window."""
        self._hits = {
            client: hits
            for client, hits in self._hits.items()
            if hits and hits[-1] > cutoff
        }

    def reset(self) -> None:
        """Forget every tracked client (test hook; also useful on reconfiguration)."""
        self._hits.clear()


_requests = _Window(60.0, "minute")
_ingests = _Window(3600.0, "hour")


def client_ip(request: Request) -> str:
    """The address to bucket this request under.

    `X-Forwarded-For` is honored only when `TRUST_PROXY_HEADERS` is set. Trusting it
    unconditionally would make the limit trivially bypassable - any client could mint
    a fresh bucket per request by varying a header it controls. When the app is the
    edge (the default), the socket peer is the only address that can't be forged.
    """
    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _sanitize_for_log(value: str) -> str:
    """Remove control characters from request-derived values before logging."""
    return "".join(c for c in value if c.isprintable() or c.isspace() and c == " ")


def _enforce(request: Request, window: _Window, limit: int) -> None:
    """Raise 429 with a `Retry-After` when `request` exceeds `window`'s limit."""
    client = client_ip(request)
    retry_after = window.hit(client, limit, _clock())
    if retry_after is None:
        return
    log.warning("rate limit exceeded",
                extra={"client": _sanitize_for_log(client),
                       "path": _sanitize_for_log(request.url.path),
                       "limit": limit})
    raise HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded ({limit} requests per {window.unit}).",
        # Always at least 1: a client told to retry after 0 seconds retries immediately.
        headers={"Retry-After": str(max(1, int(retry_after) + 1))},
    )


async def rate_limit(request: Request) -> None:
    """Dependency for the query/read endpoints (`RATE_LIMIT_PER_MINUTE`)."""
    _enforce(request, _requests, RATE_LIMIT_PER_MINUTE)


async def rate_limit_ingest(request: Request) -> None:
    """Dependency for the ingest endpoints (`RATE_LIMIT_INGEST_PER_HOUR`).

    Stacked on top of `rate_limit`, so an upload consumes a slot in both windows.
    """
    _enforce(request, _ingests, RATE_LIMIT_INGEST_PER_HOUR)


def reset() -> None:
    """Clear both windows. Keeps rate-limit state from leaking across tests."""
    _requests.reset()
    _ingests.reset()
