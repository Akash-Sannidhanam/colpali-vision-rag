"""Tests for the API-key gate (src.auth) and its wiring into the serving layer.

Patches `auth.API_KEY` - the by-value global as imported into the module that reads it -
not `src.config.API_KEY`, which nothing consults at request time. Same trap CLAUDE.md
calls out for `vector_store.QDRANT_URL`.

Note that `server.api` captured the `require_api_key` *function object* when the router
was constructed at import, so monkeypatching `server.require_api_key` would do nothing.
The key is read from the module global on every call, which is what makes these tests
possible without reimporting the server.
"""

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from src import auth, ratelimit, server

KEY = "s3cret-deploy-key"

# Endpoints reachable without a key, and why. Everything else must be gated.
#   /health - orchestrators probe liveness without holding the secret
#   /       - the UI shell itself; the API calls it makes are gated
EXEMPT_PATHS = {"/health", "/"}


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
    ratelimit.reset()                       # no bleed-through from a previous test
    with TestClient(server.app) as client:
        yield client


@pytest.fixture
def keyed(monkeypatch):
    """Turn auth on for the duration of a test."""
    monkeypatch.setattr(auth, "API_KEY", KEY)


# --- The default: no key configured, nothing gated ---

def test_auth_is_off_when_no_key_is_configured(warm):
    """The dev default. An unset API_KEY must not break a local run or the test suite."""
    assert not auth.auth_enabled()
    assert warm.get("/corpus").status_code == 200


# --- With a key configured ---

def test_missing_header_is_rejected(warm, keyed):
    """No X-API-Key at all -> 401, not a 500 or a silent pass."""
    resp = warm.get("/corpus")
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid or missing X-API-Key."


def test_wrong_key_is_rejected(warm, keyed):
    """A key that isn't the configured one -> 401."""
    assert warm.get("/corpus", headers={"X-API-Key": "wrong"}).status_code == 401


def test_correct_key_is_accepted(warm, keyed):
    """The happy path: the configured key opens the gated endpoints."""
    assert warm.get("/corpus", headers={"X-API-Key": KEY}).status_code == 200


def test_key_comparison_is_not_a_prefix_match(warm, keyed):
    """A prefix of the real key must not pass - guards against a truncating compare."""
    assert warm.get("/corpus", headers={"X-API-Key": KEY[:-1]}).status_code == 401
    assert warm.get("/corpus", headers={"X-API-Key": KEY + "x"}).status_code == 401


# --- The exemptions, stated as tests so a future change has to argue with them ---

def test_health_is_reachable_without_a_key(warm, keyed):
    """Orchestrators probe /health; requiring the secret there breaks liveness checks."""
    assert warm.get("/health").status_code == 200


def test_images_are_reachable_without_a_key(warm, keyed, monkeypatch, tmp_path):
    """The documented hole: <img src> cannot send a header, so /images stays open."""
    (tmp_path / "page.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(server, "PAGE_IMAGES_DIR", tmp_path)
    with TestClient(server.app) as client:
        # The mount was built at import against the real dir; assert the route exists and
        # is not gated, which is the property under test.
        resp = client.get("/images/does-not-exist.png")
    assert resp.status_code == 404          # StaticFiles' own 404, not the gate's 401


# --- The structural guard: gating is a property of the router, not of memory ---
#
# Stated as two halves, because the convention has two halves: routes hanging directly
# off `app` must be on the exempt list, and everything on the `api` router must resolve
# the gate. A hand-written list of paths would pass happily the day someone adds an
# unprotected endpoint; asking the app's own route table cannot.

def test_no_ungated_endpoint_hangs_directly_off_the_app():
    """`@app.get(...)` on a new endpoint fails here - it belongs on the `api` router.

    FastAPI keeps an included router as one opaque node in `app.routes`, so the APIRoutes
    left at this level are exactly the ones registered directly on `app` - which is
    precisely the mistake this guards against.
    """
    direct = [
        f"{sorted(route.methods)} {route.path}"
        for route in server.app.routes
        if isinstance(route, APIRoute)
        and route.path not in EXEMPT_PATHS
        and not route.path.startswith(("/openapi", "/docs", "/redoc"))
    ]
    assert direct == [], (
        f"these endpoints bypass the gate: {direct}. Register new routes on the `api` "
        "router in src/server.py, not on `app`."
    )


def test_the_api_router_carries_the_gate():
    """The router-level dependency is what makes every endpoint on it gated by default."""
    assert auth.require_api_key in [d.dependency for d in server.api.dependencies]


def test_every_route_on_the_api_router_resolves_the_gate():
    """And the dependency actually reaches each route, rather than only the router."""
    routes = [r for r in server.api.routes if isinstance(r, APIRoute)]
    ungated = [
        f"{sorted(r.methods)} {r.path}" for r in routes
        if auth.require_api_key not in [d.call for d in r.dependant.dependencies]
    ]
    assert ungated == []
    # Non-empty check: without it the assertion above passes vacuously if the router
    # is ever emptied or renamed out from under this test.
    assert {"/query", "/heatmap", "/ingest", "/ingest/stream", "/corpus", "/corpus/{pdf}"} == {
        r.path for r in routes
    }


def test_ingest_stacks_the_hourly_limit_on_top_of_the_per_minute_one():
    """An upload costs minutes of exclusive GPU time, so it consumes both windows."""
    for path in ("/ingest", "/ingest/stream"):
        route = next(r for r in server.api.routes
                     if isinstance(r, APIRoute) and r.path == path)
        calls = [d.call for d in route.dependant.dependencies]
        assert ratelimit.rate_limit in calls
        assert ratelimit.rate_limit_ingest in calls
