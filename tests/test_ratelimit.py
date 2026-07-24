"""Tests for the per-IP rate limiter (src.ratelimit) and its wiring into the server.

Time is injected, never slept: `_Window.hit` takes `now` as an argument and the request
path reads the module-global `ratelimit._clock`, so a test can advance a whole hour
instantly. A limiter tested with real sleeps is a limiter that only gets tested once.

Config knobs are patched as imported into `ratelimit` (`ratelimit.RATE_LIMIT_PER_MINUTE`),
not on `src.config` - the by-value convention CLAUDE.md describes.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import auth, ratelimit, server


@pytest.fixture(autouse=True)
def clean_windows():
    """Rate-limit state is process-global; reset it around every test in this module."""
    ratelimit.reset()
    yield
    ratelimit.reset()


# --- The window itself (pure logic, no HTTP) ---

def test_requests_under_the_limit_are_allowed():
    """Every hit up to the limit is accepted."""
    window = ratelimit._Window(60.0, "minute")
    assert [window.hit("ip", 3, now=0.0) for _ in range(3)] == [None, None, None]


def test_the_request_past_the_limit_is_refused_with_a_wait():
    """The 4th of 3 comes back with the seconds until a slot frees up."""
    window = ratelimit._Window(60.0, "minute")
    for _ in range(3):
        window.hit("ip", 3, now=0.0)
    assert window.hit("ip", 3, now=10.0) == pytest.approx(50.0)


def test_the_window_slides():
    """Once the oldest hit ages out, the client is served again."""
    window = ratelimit._Window(60.0, "minute")
    for _ in range(3):
        window.hit("ip", 3, now=0.0)
    assert window.hit("ip", 3, now=59.0) is not None      # still inside the window
    assert window.hit("ip", 3, now=61.0) is None          # the first three have expired


def test_refused_requests_are_not_recorded():
    """Hammering a closed door must not push the recovery time out forever.

    The client recovers one window after its last *accepted* request, not after its last
    attempt - otherwise a retry loop locks itself out indefinitely.
    """
    window = ratelimit._Window(60.0, "minute")
    for _ in range(3):
        window.hit("ip", 3, now=0.0)
    for t in range(1, 60):                                # hammer throughout the window
        assert window.hit("ip", 3, now=float(t)) is not None
    assert window.hit("ip", 3, now=61.0) is None          # still recovers on schedule


def test_a_limit_of_zero_disables_the_window():
    """0 is the documented off switch."""
    window = ratelimit._Window(60.0, "minute")
    assert all(window.hit("ip", 0, now=0.0) is None for _ in range(1000))


def test_clients_get_independent_buckets():
    """One noisy address must not exhaust another's allowance."""
    window = ratelimit._Window(60.0, "minute")
    for _ in range(3):
        window.hit("noisy", 3, now=0.0)
    assert window.hit("noisy", 3, now=0.0) is not None
    assert window.hit("quiet", 3, now=0.0) is None


def test_idle_clients_are_swept_so_the_map_stays_bounded():
    """A spray of one-request-each addresses must not grow the map without bound."""
    window = ratelimit._Window(60.0, "minute")
    for i in range(ratelimit._SWEEP_AT + 50):
        window.hit(f"ip-{i}", 10, now=0.0)
    # Every entry is still live at t=0, so nothing is swept yet...
    assert len(window._hits) == ratelimit._SWEEP_AT + 50
    # ...but one hit after they have all aged out collects them.
    window.hit("fresh", 10, now=120.0)
    assert len(window._hits) == 1


# --- client_ip: which address a request is bucketed under ---

def _fake_request(peer: str = "10.0.0.1", forwarded: str | None = None):
    """A stand-in exposing just the two attributes client_ip reads."""
    headers = {"x-forwarded-for": forwarded} if forwarded else {}
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=peer))


def test_forwarded_header_is_ignored_by_default(monkeypatch):
    """The security-relevant default: a forged header must not mint a fresh bucket."""
    monkeypatch.setattr(ratelimit, "TRUST_PROXY_HEADERS", False)
    assert ratelimit.client_ip(_fake_request(forwarded="1.2.3.4")) == "10.0.0.1"


def test_forwarded_header_is_honored_when_explicitly_trusted(monkeypatch):
    """Behind a real proxy the socket peer is the proxy, so the header is the client."""
    monkeypatch.setattr(ratelimit, "TRUST_PROXY_HEADERS", True)
    assert ratelimit.client_ip(_fake_request(forwarded="1.2.3.4")) == "1.2.3.4"


def test_the_first_hop_of_a_forwarded_chain_wins(monkeypatch):
    """X-Forwarded-For accumulates hops left-to-right; the original client is first."""
    monkeypatch.setattr(ratelimit, "TRUST_PROXY_HEADERS", True)
    assert ratelimit.client_ip(_fake_request(forwarded="1.2.3.4, 10.1.1.1")) == "1.2.3.4"


def test_a_missing_peer_falls_back_rather_than_crashing(monkeypatch):
    """request.client is None for some ASGI transports; bucket them together."""
    monkeypatch.setattr(ratelimit, "TRUST_PROXY_HEADERS", False)
    assert ratelimit.client_ip(SimpleNamespace(headers={}, client=None)) == "unknown"


# --- Through the actual HTTP stack ---

@pytest.fixture
def warm(monkeypatch):
    """Stub every lifespan seam so startup runs nothing heavy; yield a live TestClient."""
    monkeypatch.setattr(server, "validate", lambda: None)
    monkeypatch.setattr(server, "load_model", lambda: None)
    monkeypatch.setattr(server, "get_graph", lambda: None)
    monkeypatch.setattr(server, "ping", lambda: None)
    monkeypatch.setattr(server, "is_loaded", lambda: True)
    monkeypatch.setattr(server, "close_client", lambda: None)
    monkeypatch.setattr(server, "list_documents", lambda: [])
    with TestClient(server.app) as client:
        yield client


def test_endpoint_returns_429_once_the_limit_is_hit(warm, monkeypatch):
    """Three allowed, the fourth refused with a Retry-After the client can act on."""
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_PER_MINUTE", 3)
    monkeypatch.setattr(ratelimit, "_clock", lambda: 0.0)

    assert [warm.get("/corpus").status_code for _ in range(3)] == [200, 200, 200]

    refused = warm.get("/corpus")
    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) >= 1
    assert "per minute" in refused.json()["detail"]


def test_the_client_recovers_after_the_window(warm, monkeypatch):
    """The limit throttles; it does not permanently ban."""
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_PER_MINUTE", 2)
    now = 0.0
    monkeypatch.setattr(ratelimit, "_clock", lambda: now)

    assert warm.get("/corpus").status_code == 200
    assert warm.get("/corpus").status_code == 200
    assert warm.get("/corpus").status_code == 429

    now = 61.0
    assert warm.get("/corpus").status_code == 200


def test_health_is_never_rate_limited(warm, monkeypatch):
    """Throttling the liveness probe would make a busy server look dead to its orchestrator."""
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_PER_MINUTE", 1)
    monkeypatch.setattr(ratelimit, "_clock", lambda: 0.0)
    assert all(warm.get("/health").status_code == 200 for _ in range(20))


def test_zero_disables_limiting_end_to_end(warm, monkeypatch):
    """The documented off switch, exercised through the real dependency chain."""
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(ratelimit, "_clock", lambda: 0.0)
    assert all(warm.get("/corpus").status_code == 200 for _ in range(50))


def test_unauthenticated_requests_are_throttled_too(warm, monkeypatch):
    """The limiter must run BEFORE the API-key gate, not after.

    FastAPI solves router dependencies in order. With auth first, a caller holding no
    key burns unlimited 401s - the throttle would only ever apply to requests that had
    already passed the gate, leaving key guessing and unauthenticated floods unbounded.
    Caught on a live server; this pins the ordering so it can't silently flip back.
    """
    monkeypatch.setattr(auth, "API_KEY", "the-real-key")
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_PER_MINUTE", 3)
    monkeypatch.setattr(ratelimit, "_clock", lambda: 0.0)

    codes = [warm.get("/corpus", headers={"X-API-Key": "guess"}).status_code
             for _ in range(6)]

    assert codes[:3] == [401, 401, 401]          # rejected, but counted
    assert codes[3:] == [429, 429, 429]          # then throttled like anything else
