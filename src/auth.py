"""Shared-secret API-key auth for the serving layer.

One dependency, `require_api_key`, attached to the protected router in `src/server.py`
rather than to individual routes - so an endpoint added later is gated by default
instead of by remembering to decorate it.

Two deliberate holes, both documented in the README:

  * `/health` is open, so a container orchestrator can probe liveness without holding
    the secret.
  * The `/images` static mount is open, because `<img src>` cannot send a custom
    header. Page and crop PNGs are the only thing it exposes, and their paths are
    discoverable only through an authenticated `/query`.

This is deployment gating - one shared secret for whoever you hand it to - not user
identity. There are no accounts, sessions, or scopes.
"""

import secrets

from fastapi import Header, HTTPException

from src.config import API_KEY
from src.logging_setup import get_logger

log = get_logger("auth")


def auth_enabled() -> bool:
    """Whether a key is configured. False (auth off) is the dev/test default."""
    return bool(API_KEY)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject a request whose `X-API-Key` header doesn't match the configured key.

    A no-op when `API_KEY` is unset, which keeps local runs and the test suite working
    with no configuration. The comparison is `secrets.compare_digest`, not `==`, so a
    wrong key takes the same time to reject regardless of how many leading characters
    it got right.
    """
    if not API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        # Log the rejection but never the submitted value - it lands in log storage.
        log.warning("rejected unauthenticated request", extra={"has_header": bool(x_api_key)})
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")
