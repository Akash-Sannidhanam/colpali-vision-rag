"""Tests for the FastAPI serving layer (src.server).

Stubs the pipeline seams as imported into `server` - run_query, run_ingest, ping,
validate, load_model, get_graph, is_loaded, list_documents - so the TestClient
exercises routing, the response contract, and image URL/base64 shaping with no models,
network, API key, or real Qdrant. Lifespan runs against the stubs via the
`with TestClient(...)` context-manager form (its startup calls the stubbed
validate/load_model/ping/get_graph, never real weights).

Because the endpoint handlers resolve module globals (`ping`, `run_query`,
`MAX_UPLOAD_MB`, `_encode_data_uri`, ...) at call time, a test can monkeypatch them
*after* the client is warm to drive a specific per-request path (e.g. flip `ping` to
raise for the degraded-health case without breaking startup).
"""

import asyncio
import base64
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from src import ratelimit, server


def _canned_result(found: bool = True) -> dict:
    """A run_query-shaped RAGState dict: two reranked pages + a meta block with stages."""
    retrieved = [
        {"pdf": "attention.pdf", "page_number": 3, "score": 14.2,
         "image_path": str(server.PAGE_IMAGES_DIR / "attention_page_3.png")},
        {"pdf": "attention.pdf", "page_number": 5, "score": 8.1,
         "image_path": str(server.PAGE_IMAGES_DIR / "attention_page_5.png")},
    ]
    crops = server.PAGE_IMAGES_DIR / "crops"
    if found:
        citation = {"answer": "42", "found": True, "source_page": 1, "box": [140, 300, 660, 700],
                    "confidence": "high"}
        crop = str(crops / "attention_page_3_crop_0.png")
        annotated = str(crops / "attention_page_3_annotated.png")
        # two cited regions across the two reranked pages, each with its own crop
        cited_regions = [
            {"source_page": 1, "box": [140, 300, 660, 700], "crop_path": crop},
            {"source_page": 2, "box": [100, 100, 400, 400],
             "crop_path": str(crops / "attention_page_5_crop_1.png")},
        ]
    else:
        # confidence intentionally omitted -> the server should fall back to "low".
        citation = {"answer": "Couldn't find it.", "found": False, "source_page": 0, "box": []}
        crop = annotated = None
        cited_regions = []
    return {
        "question": "what?",
        "retrieved": retrieved,
        "answer": citation["answer"],
        "citation": citation,
        "crop_path": crop,
        "annotated_path": annotated,
        "cited_regions": cited_regions,
        "meta": {
            "request_id": "abc123", "latency_ms": 3400.0,
            "prompt_tokens": 9000, "output_tokens": 2000, "total_tokens": 11000,
            "est_cost_usd": 0.015, "gemini_calls": 2,
            "retrieval_confidence": 0.82 if found else None,
            "stages": [
                {"node": "retrieve", "latency_ms": 900.0, "total_tokens": 0, "est_cost_usd": 0.0, "gemini_calls": 0},
                {"node": "rerank", "latency_ms": 1200.0, "total_tokens": 3100, "est_cost_usd": 0.004, "gemini_calls": 1},
                {"node": "answer", "latency_ms": 1000.0, "total_tokens": 7900, "est_cost_usd": 0.011, "gemini_calls": 1},
                {"node": "highlight", "latency_ms": 300.0, "total_tokens": 0, "est_cost_usd": 0.0, "gemini_calls": 0},
            ],
        },
    }


@pytest.fixture
def warm(monkeypatch):
    """Stub every lifespan seam so startup runs nothing heavy; yield a live TestClient."""
    # The limiter is a process-global sliding window keyed on client IP, and every test
    # here is the same "testclient" address - so without this the *suite* is the client,
    # and the 31st request in any 60s window 429s whichever test happens to make it.
    # test_auth and test_ratelimit already reset in their fixtures; this file did not, and
    # was sitting far enough under RATE_LIMIT_PER_MINUTE that it only surfaced when a batch
    # of new tests was added.
    ratelimit.reset()
    monkeypatch.setattr(server, "validate", lambda: None)
    monkeypatch.setattr(server, "load_model", lambda: None)
    monkeypatch.setattr(server, "get_graph", lambda: None)
    monkeypatch.setattr(server, "ping", lambda: None)
    monkeypatch.setattr(server, "is_loaded", lambda: True)
    monkeypatch.setattr(server, "close_client", lambda: None)
    # A whole corpus by default (lifespan checks it, and /health reports it); the
    # half-restored case is driven per-test by re-patching this.
    monkeypatch.setattr(server, "index_health",
                        lambda: {"ok": True, "checked": 0, "incomplete": []})
    with TestClient(server.app) as client:
        yield client


# --- /health ---

def test_health_ok(warm):
    """A warm server with a reachable Qdrant and a whole corpus reports ok."""
    resp = warm.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True,
                           "qdrant": "ok", "corpus": "ok"}


def test_health_degraded_when_qdrant_unreachable(warm, monkeypatch):
    """A Qdrant blip after boot reports 503 degraded, surfacing the reason."""
    def down():
        """Simulate Qdrant being unreachable after a successful boot."""
        raise RuntimeError("Cannot reach Qdrant")

    monkeypatch.setattr(server, "ping", down)          # after startup, so boot still succeeded
    resp = warm.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert "Cannot reach Qdrant" in body["qdrant"]


# --- /corpus ---

def test_corpus_lists_documents_and_total(warm, monkeypatch):
    """The rail's payload lists each document and the summed page count."""
    monkeypatch.setattr(server, "list_documents",
                        lambda: [{"pdf": "attention.pdf", "page_count": 15},
                                 {"pdf": "colpali.pdf", "page_count": 26}])
    resp = warm.get("/corpus")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_pages"] == 41
    assert body["qdrant"] == "ok"
    assert [d["pdf"] for d in body["documents"]] == ["attention.pdf", "colpali.pdf"]


# --- DELETE /corpus/{pdf} ---

@pytest.fixture
def corpus_files(monkeypatch, tmp_path):
    """A tmp corpus on disk: two documents' pages + crops, plus their source PDFs."""
    pages, crops, pdfs = tmp_path / "page_images", tmp_path / "page_images" / "crops", tmp_path / "pdfs"
    for d in (pages, crops, pdfs):
        d.mkdir(parents=True)
    written: dict[str, list] = {}
    for name in ("doomed.pdf", "keeper.pdf"):
        stem = name[:-4]
        written[name] = [
            pages / f"{stem}_page_1.png", pages / f"{stem}_page_2.png",
            crops / f"{stem}_page_1_crop_0.png", crops / f"{stem}_page_1_annotated.png",
            pdfs / name,
        ]
        for p in written[name]:
            p.write_bytes(b"x")

    monkeypatch.setattr(server, "PAGE_IMAGES_DIR", pages)
    monkeypatch.setattr(server, "PDFS_DIR", pdfs)
    # the file-matching helpers resolve their directories from src.config at call time
    monkeypatch.setattr("src.pdf_render.PAGE_IMAGES_DIR", pages)
    monkeypatch.setattr("src.pdf_render.CROPS_DIR", crops)
    monkeypatch.setattr(server, "document_index", lambda: {
        "doomed.pdf": {"page_count": 2, "content_hash": "h", "embed_version": "m"},
        "keeper.pdf": {"page_count": 2, "content_hash": "h", "embed_version": "m"},
    })
    monkeypatch.setattr(server, "delete_document", lambda name: 2)
    return written


def test_delete_removes_vectors_and_every_file_for_that_document(warm, corpus_files, monkeypatch):
    """Delete drops the vectors plus every page, crop, and the source PDF - and nothing belonging to another document."""
    dropped: list[str] = []
    monkeypatch.setattr(server, "delete_document", lambda name: dropped.append(name) or 2)

    resp = warm.delete("/corpus/doomed.pdf")

    assert resp.status_code == 200
    assert resp.json() == {"pdf": "doomed.pdf", "removed_pages": 2}
    assert dropped == ["doomed.pdf"]
    assert not any(p.exists() for p in corpus_files["doomed.pdf"])   # pages, crops, PDF
    assert all(p.exists() for p in corpus_files["keeper.pdf"])       # the other doc is intact


def test_delete_404s_for_an_unindexed_document_without_touching_disk(warm, corpus_files, monkeypatch):
    """An unknown document 404s before any vector or file is touched."""
    dropped: list[str] = []
    monkeypatch.setattr(server, "delete_document", lambda name: dropped.append(name) or 0)

    resp = warm.delete("/corpus/ghost.pdf")

    assert resp.status_code == 404
    assert dropped == []                                             # short-circuits first
    assert all(p.exists() for p in corpus_files["keeper.pdf"])


@pytest.mark.parametrize("target", ["..%2F..%2Fkeeper.pdf", "..%2F..%2Fetc%2Fpasswd",
                                    "%2Fetc%2Fpasswd"])
def test_delete_path_traversal_never_reaches_the_handler(warm, corpus_files, monkeypatch, target):
    """First line of defence: a `{pdf}` path parameter does not match `/`, so a name
    carrying a separator 404s in the router before any of our code runs."""
    dropped: list[str] = []
    monkeypatch.setattr(server, "delete_document", lambda name: dropped.append(name) or 0)

    resp = warm.delete(f"/corpus/{target}")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Not Found"          # the router's, not ours
    assert dropped == []
    assert all(p.exists() for p in corpus_files["keeper.pdf"])


def test_delete_only_ever_touches_an_indexed_document(warm, corpus_files, monkeypatch, tmp_path):
    """A PDF in PDFS_DIR that was never ingested is not the API's to delete."""
    # Second line of defence, and the one that actually constrains the filesystem: the
    # normalized name must be in the index before anything is unlinked. A PDF sitting in
    # PDFS_DIR that was never ingested is not the API's to delete.
    stray = server.PDFS_DIR / "never_ingested.pdf"
    stray.write_bytes(b"%PDF-1.4")
    unlinked: list[str] = []
    monkeypatch.setattr(server, "_remove_document",
                        lambda name: unlinked.append(name) or 0)

    assert warm.delete("/corpus/never_ingested.pdf").status_code == 404

    assert unlinked == []
    assert stray.exists()


def test_delete_normalizes_the_name_before_looking_it_up(warm, corpus_files, monkeypatch):
    """The name checked against the index is the same one that gets deleted."""
    # `Path(pdf).name` runs before the index lookup, so the name that is checked is the
    # same one that is deleted - a lookup on a raw string and an unlink on a normalized
    # one would be exactly the mismatch that lets something unintended through.
    looked_up: list[str] = []
    monkeypatch.setattr(server, "_remove_document", lambda name: looked_up.append(name) or 2)

    assert warm.delete("/corpus/doomed.pdf").status_code == 200
    assert looked_up == ["doomed.pdf"]


def test_delete_survives_a_missing_file_on_disk(warm, corpus_files):
    """File removal is best-effort: an already-missing page doesn't fail the request."""
    # Files are best-effort: a page image already gone must not fail the request after
    # the vectors have been removed.
    corpus_files["doomed.pdf"][0].unlink()

    resp = warm.delete("/corpus/doomed.pdf")

    assert resp.status_code == 200
    assert not any(p.exists() for p in corpus_files["doomed.pdf"])


# --- /query ---

def test_query_happy_path_shape(warm, monkeypatch):
    """The full /query contract: enriched citation, per-region crops, candidate pages, and meta."""
    monkeypatch.setattr(server, "run_query", lambda q: _canned_result(found=True))
    resp = warm.post("/query", json={"question": "what was the Q4 revenue?"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"] == "42"
    cit = body["citation"]
    assert cit["found"] is True and cit["source_page"] == 1
    assert cit["box"] == [140, 300, 660, 700]
    assert cit["pdf"] == "attention.pdf" and cit["page_number"] == 3   # enriched from pages[0]
    assert cit["confidence"] == "high"                                 # model self-report

    # multi-region: each cited region carries its own page enrichment + crop image
    assert len(cit["regions"]) == 2
    assert cit["regions"][0]["source_page"] == 1 and cit["regions"][0]["page_number"] == 3
    assert cit["regions"][1]["source_page"] == 2 and cit["regions"][1]["page_number"] == 5
    assert cit["regions"][0]["crop"]["url"].endswith("attention_page_3_crop_0.png")
    assert cit["regions"][1]["crop"]["url"].endswith("attention_page_5_crop_1.png")

    assert len(body["pages"]) == 2
    assert body["pages"][0]["image"]["url"].endswith("/images/attention_page_3.png")
    assert body["pages"][0]["image"]["data_uri"] is None               # url mode by default

    assert body["crop"]["url"].endswith("attention_page_3_crop_0.png")
    assert body["annotated"]["url"].endswith("attention_page_3_annotated.png")

    meta = body["meta"]
    assert meta["request_id"] == "abc123"
    assert meta["total_tokens"] == 11000 and meta["est_cost_usd"] == 0.015
    assert meta["retrieve_k"] == server.RETRIEVE_K
    assert meta["retrieval_confidence"] == 0.82                        # deterministic retrieval signal
    assert [s["node"] for s in meta["stages"]] == ["retrieve", "rerank", "answer", "highlight"]


def test_query_not_found_has_no_crop(warm, monkeypatch):
    """A not-found answer has no crop or regions but still surfaces the candidates."""
    monkeypatch.setattr(server, "run_query", lambda q: _canned_result(found=False))
    resp = warm.post("/query", json={"question": "unanswerable?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citation"]["found"] is False and body["citation"]["source_page"] == 0
    assert body["citation"]["pdf"] is None
    assert body["citation"]["confidence"] == "low"                     # default when omitted
    assert body["citation"]["regions"] == []                           # no regions when not found
    assert body["meta"]["retrieval_confidence"] is None                # nothing cited
    assert body["crop"] is None and body["annotated"] is None
    assert len(body["pages"]) == 2                                     # candidates still surfaced


def test_query_inline_populates_data_uri(warm, monkeypatch):
    """?inline=true adds base64 data-URIs while keeping the static URLs."""
    monkeypatch.setattr(server, "run_query", lambda q: _canned_result(found=True))

    async def fake_encode(p):  # _encode_data_uri is async (off-loop file read)
        """Stand in for the async data-URI encoder without touching disk."""
        return "data:image/png;base64,STUB"

    monkeypatch.setattr(server, "_encode_data_uri", fake_encode)
    resp = warm.post("/query?inline=true", json={"question": "q"})
    body = resp.json()
    assert body["pages"][0]["image"]["data_uri"] == "data:image/png;base64,STUB"
    assert body["pages"][0]["image"]["url"] is not None                # url stays populated too
    assert body["crop"]["data_uri"] == "data:image/png;base64,STUB"


def test_query_empty_question_is_422(warm):
    """An empty or missing question is rejected by validation."""
    assert warm.post("/query", json={"question": ""}).status_code == 422
    assert warm.post("/query", json={}).status_code == 422


# --- /heatmap ---

def test_heatmap_happy_path(warm, monkeypatch, tmp_path):
    """The heatmap endpoint returns the patch grid with its n_x/n_y dimensions."""
    png = tmp_path / "attention_page_3.png"
    png.write_bytes(b"x")                                  # only existence is checked here
    monkeypatch.setattr(server, "page_image_path", lambda pdf, page: png)
    monkeypatch.setattr(server, "page_similarity",
                        lambda q, path: ([[0.0, 1.0], [0.5, 0.25]], 2, 2))
    resp = warm.post("/heatmap",
                     json={"question": "where's the total?", "pdf": "attention.pdf", "page_number": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pdf"] == "attention.pdf" and body["page_number"] == 3
    assert body["n_x"] == 2 and body["n_y"] == 2
    assert body["grid"] == [[0.0, 1.0], [0.5, 0.25]]       # n_y rows x n_x cols, grid[y][x]


def test_heatmap_404_when_page_missing_never_runs_model(warm, monkeypatch, tmp_path):
    """An unindexed page 404s before paying for the GPU forward passes."""
    monkeypatch.setattr(server, "page_image_path", lambda pdf, page: tmp_path / "nope.png")
    ran = {"model": False}

    def spy(q, path):
        """Record whether the model was reached, and return an empty grid."""
        ran["model"] = True
        return ([], 0, 0)

    monkeypatch.setattr(server, "page_similarity", spy)
    resp = warm.post("/heatmap", json={"question": "q", "pdf": "ghost.pdf", "page_number": 9})
    assert resp.status_code == 404
    assert ran["model"] is False                           # short-circuits before the GPU work


def test_heatmap_validates_request(warm):
    """Blank questions, non-positive pages, and missing fields are rejected."""
    assert warm.post("/heatmap",
                     json={"question": "", "pdf": "a.pdf", "page_number": 1}).status_code == 422
    assert warm.post("/heatmap",
                     json={"question": "q", "pdf": "a.pdf", "page_number": 0}).status_code == 422
    assert warm.post("/heatmap", json={"question": "q", "pdf": "a.pdf"}).status_code == 422


# --- /ingest ---

def test_ingest_happy_path(warm, monkeypatch, tmp_path):
    """An upload is saved under PDFS_DIR and handed to the ingest by path."""
    captured = {}

    def fake_run_ingest(paths, *, gate=None):
        """Record what the endpoint handed the ingest, and report a page count."""
        captured["paths"] = paths
        captured["gate"] = gate
        return 7

    monkeypatch.setattr(server, "PDFS_DIR", tmp_path)                  # don't write into the repo's pdfs/
    monkeypatch.setattr(server, "run_ingest", fake_run_ingest)
    resp = warm.post("/ingest", files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert resp.status_code == 200
    assert resp.json() == {"pdf": "doc.pdf", "indexed_pages": 7}
    assert (tmp_path / "doc.pdf").read_bytes() == b"%PDF-1.4 fake"     # saved under PDFS_DIR
    assert captured["paths"] == [tmp_path / "doc.pdf"]                 # run_ingest got the saved path
    # Without a gate the ingest holds the model for the whole document, which is the
    # head-of-line blocking this endpoint used to impose on every concurrent query.
    assert callable(captured["gate"])


def test_ingest_does_not_re_embed_the_rest_of_the_corpus(warm, monkeypatch, tmp_path):
    """Only the uploaded PDF is ingested - adding one document no longer re-embeds the corpus."""
    # An upload used to hand run_ingest every PDF in PDFS_DIR, so adding one document
    # re-rendered and re-embedded the whole corpus through the ~2B model.
    (tmp_path / "already_indexed.pdf").write_bytes(b"%PDF-1.4 old")
    captured = {}
    monkeypatch.setattr(server, "PDFS_DIR", tmp_path)
    monkeypatch.setattr(server, "run_ingest",
                        lambda paths, *, gate=None: captured.update(paths=paths) or 2)

    warm.post("/ingest", files={"file": ("new.pdf", b"%PDF-1.4 new", "application/pdf")})

    assert captured["paths"] == [tmp_path / "new.pdf"]                 # only the upload


def test_ingest_rejects_non_pdf(warm):
    """A non-PDF upload is rejected before anything is written."""
    resp = warm.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_ingest_stream_emits_progress_and_done(warm, monkeypatch, tmp_path):
    """The SSE endpoint streams per-page progress and a final done frame."""
    monkeypatch.setattr(server, "PDFS_DIR", tmp_path)

    def fake_run_ingest(paths, progress, *, gate=None):
        """Record what the endpoint handed the ingest, and report a page count."""
        assert callable(gate)          # the SSE path yields the model between batches too
        progress({"phase": "render", "pdf": "doc.pdf"})
        progress({"phase": "embed", "pdf": "doc.pdf", "page": 1, "total": 1})
        return 1

    monkeypatch.setattr(server, "run_ingest", fake_run_ingest)
    resp = warm.post("/ingest/stream",
                     files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    frames = resp.text
    assert '"phase": "render"' in frames
    assert '"phase": "embed"' in frames                    # per-page progress streamed
    assert '"phase": "done"' in frames and '"indexed_pages": 1' in frames


def test_ingest_stream_rejects_non_pdf(warm):
    """Upload validation happens before the stream opens, so a bad upload is a plain 400."""
    resp = warm.post("/ingest/stream", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert resp.status_code == 400


def test_ingest_rejects_oversize(warm, monkeypatch):
    """An upload over the size cap is rejected with 413."""
    monkeypatch.setattr(server, "MAX_UPLOAD_MB", 0)                    # anything non-empty is too big
    resp = warm.post("/ingest", files={"file": ("big.pdf", b"%PDF-1.4 xxxxxxxx", "application/pdf")})
    assert resp.status_code == 413


# --- pure image helpers (no server) ---

def test_to_url_maps_page_images_path(monkeypatch, tmp_path):
    """A filesystem path under page_images/ maps to its /images/... URL."""
    monkeypatch.setattr(server, "PAGE_IMAGES_DIR", tmp_path)
    fs = tmp_path / "crops" / "sales_report_page_1_crop.png"
    fs.parent.mkdir(parents=True)
    fs.write_bytes(b"x")
    url = server._to_url(SimpleNamespace(base_url="http://testserver/"), str(fs))
    assert url == "http://testserver/images/crops/sales_report_page_1_crop.png"


def test_encode_data_uri_roundtrips(tmp_path):
    """The data-URI encoder round-trips the exact file bytes."""
    fs = tmp_path / "p.png"
    payload = b"\x89PNG\r\n fake bytes"
    fs.write_bytes(payload)
    uri = asyncio.run(server._encode_data_uri(str(fs)))  # helper is async
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == payload


# --- serving the built UI (ui/dist) ---
#
# ui/dist is gitignored, so CI only ever sees the not-built case. `_mount_ui` takes the
# app to mount onto precisely so both branches are reachable from a test: point
# UI_DIST_DIR at a tmp_path and mount onto a throwaway app.

@pytest.fixture
def built_dist(monkeypatch, tmp_path):
    """A minimal ui/dist: index.html plus one hashed asset."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>Vision RAG</title>")
    (tmp_path / "assets" / "index-abc123.js").write_text("console.log(1)")
    monkeypatch.setattr(server, "UI_DIST_DIR", tmp_path)
    return tmp_path


def test_ui_is_not_mounted_when_it_has_not_been_built(monkeypatch, tmp_path):
    """A dev checkout that never ran `npm run build` must still serve the API.

    StaticFiles raises on a missing directory, so an unguarded mount would take the
    whole API down over a frontend that was never built.
    """
    monkeypatch.setattr(server, "UI_DIST_DIR", tmp_path / "never-built")
    fresh = FastAPI()
    assert server._mount_ui(fresh) is False
    assert [r.path for r in fresh.routes if isinstance(r, APIRoute)] == []


def test_an_interrupted_build_counts_as_not_built(monkeypatch, tmp_path):
    """An empty ui/dist (interrupted build) is not a servable UI - index.html decides."""
    (tmp_path / "assets").mkdir()
    monkeypatch.setattr(server, "UI_DIST_DIR", tmp_path)          # no index.html
    assert server._mount_ui(FastAPI()) is False


def test_built_ui_is_served_at_the_root(built_dist):
    """The deployed shape: the UI and the API share one origin."""
    fresh = FastAPI()
    assert server._mount_ui(fresh) is True
    with TestClient(fresh) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "Vision RAG" in index.text
        assert client.get("/assets/index-abc123.js").status_code == 200


def test_serving_the_ui_does_not_shadow_the_api(built_dist, warm):
    """The reason this is an explicit GET route and not `mount("/", html=True)`.

    A catch-all mount matches every path before the method is considered, which turns
    an unrouted DELETE into StaticFiles' 405 - and the path-traversal guard on
    DELETE /corpus/{pdf} depends on getting the router's 404 instead.
    """
    # Snapshot/restore rather than filtering by route name: ui/dist exists on a dev box
    # (so the import-time mount is already there) but not in CI, and only restoring the
    # exact prior list leaves server.app - a module global - untouched either way.
    original_routes = list(server.app.router.routes)
    try:
        server._mount_ui(server.app)                    # mount onto the *live* app
        assert warm.delete("/corpus/..%2F..%2Fetc%2Fpasswd").status_code == 404
        assert warm.get("/health").status_code == 200
        assert warm.get("/images/nope.png").status_code == 404
        assert warm.get("/").status_code == 200         # and the UI still serves
    finally:
        server.app.router.routes = original_routes


# --- load shedding + corpus integrity ---

@pytest.mark.parametrize("path,body", [
    ("/query", {"question": "what?"}),
    ("/heatmap", {"question": "what?", "pdf": "a.pdf", "page_number": 1}),
])
def test_gpu_busy_is_shed_as_503_with_retry_after(warm, monkeypatch, path, body):
    """A saturated server answers 503 + Retry-After rather than hanging the client.

    The arbiter's own shedding is covered in test_gpu_arbiter; what is server-side here
    is the mapping - a GpuBusy escaping any GPU endpoint must reach the client as a
    retryable 503, the same shape ratelimit gives a 429, not a bare 500.
    """
    async def busy(*args, **kwargs):
        raise server.GpuBusy(30.0)

    monkeypatch.setattr(server._gpu, "run_exclusive", busy)
    # An existing page, so /heatmap reaches the model instead of 404ing first.
    monkeypatch.setattr(server, "page_image_path", lambda pdf, n: Path(__file__))

    resp = warm.post(path, json=body)

    assert resp.status_code == 503
    assert int(resp.headers["Retry-After"]) >= 1
    assert "busy" in resp.json()["detail"].lower()


def test_health_reports_a_corpus_missing_its_page_images(warm, monkeypatch):
    """The vectors-without-images split is named on /health, not left to be inferred.

    /corpus still lists the document (it reads Qdrant payloads) and every query answers
    "not found", so without this the deployment looks entirely healthy.
    """
    monkeypatch.setattr(server, "index_health", lambda: {
        "ok": False, "checked": 2,
        "incomplete": [{"pdf": "b.pdf", "indexed_pages": 2, "images_present": 0}]})

    body = warm.get("/health").json()

    assert body["status"] == "ok"            # the server is serving; the corpus is not whole
    assert "b.pdf" in body["corpus"] and "re-run ingest" in body["corpus"]


def test_health_survives_an_index_health_failure_without_leaking_the_reason(warm, monkeypatch):
    """A broken integrity check must not take down the liveness probe - or leak internals.

    /health is registered on `app`, not the gated `api` router, so its body is world
    -readable. An exception string here can carry internal hostnames, filesystem paths or
    collection names, so the reason goes to the log and the client gets a bare status.
    """
    def boom():
        raise RuntimeError("connect failed to qdrant-internal.svc:6333")

    monkeypatch.setattr(server, "index_health", boom)
    body = warm.get("/health").json()

    assert body["status"] == "ok"
    assert body["corpus"] == "unknown"
    assert "qdrant-internal" not in body["corpus"]


def test_ingest_stream_sheds_as_503_before_the_stream_opens(warm, monkeypatch, tmp_path):
    """A shed SSE ingest must be a real 503, not a truncated 200 with no status.

    FastAPI sends a StreamingResponse's 200 headers before it ever iterates the generator,
    so acquiring the model *inside* event_stream() put GpuBusy out of reach of the
    exception handler - the client got a 200, an empty body, and no Retry-After, while
    plain /ingest sheds correctly. The two ingest endpoints must agree under load.
    """
    monkeypatch.setattr(server, "PDFS_DIR", tmp_path)

    class _BusyAcquire:
        async def __aenter__(self):
            raise server.GpuBusy(30.0)

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(server._gpu, "acquire", lambda *a, **k: _BusyAcquire())
    monkeypatch.setattr(server, "run_ingest",
                        lambda *a, **k: pytest.fail("the build must never start"))

    resp = warm.post("/ingest/stream",
                     files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")})

    assert resp.status_code == 503
    assert int(resp.headers["Retry-After"]) >= 1
    assert "text/event-stream" not in resp.headers.get("content-type", "")


def test_ingest_stream_releases_the_model_when_the_build_finishes(warm, monkeypatch, tmp_path):
    """The lock is now taken in the endpoint, so the generator must still give it back."""
    monkeypatch.setattr(server, "PDFS_DIR", tmp_path)
    monkeypatch.setattr(server, "run_ingest", lambda paths, progress, *, gate=None: 1)

    resp = warm.post("/ingest/stream",
                     files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert resp.status_code == 200
    assert '"phase": "done"' in resp.text

    # A leaked lock would make every subsequent request hang or shed.
    assert not server._gpu._lock.locked()
    assert not server._gpu.contended()


def test_ingest_stream_never_holds_the_model_across_the_response_boundary(
        warm, monkeypatch, tmp_path):
    """The lock must not survive a connection that breaks before the headers go out.

    Holding it in the endpoint and closing the context *inside* the generator looks
    equivalent and is not: when send() fails on the first message Starlette never starts
    the body at all, so its `finally` never runs and the model stays locked for the life
    of the process — every later request sheds forever. That is strictly worse than the
    truncated-200 bug it was fixing, so acquire and release live in the same frame.
    """
    monkeypatch.setattr(server, "PDFS_DIR", tmp_path)
    monkeypatch.setattr(server, "run_ingest", lambda paths, progress, *, gate=None: 1)

    body = (b'--b\r\nContent-Disposition: form-data; name="file"; filename="doc.pdf"\r\n'
            b"Content-Type: application/pdf\r\n\r\n%PDF-1.4 fake\r\n--b--\r\n")

    async def drive():
        """Call the ASGI app directly with a send() that fails on the FIRST message.

        That is the abandoned-stream shape: Starlette raises out of `http.response.start`
        and never advances the body generator, so any cleanup written in the generator
        never runs. TestClient cannot produce this — it always consumes the body.
        """
        async def send(message):
            raise OSError("client vanished before the headers went out")

        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        scope = {
            "type": "http", "http_version": "1.1", "method": "POST",
            "path": "/ingest/stream", "raw_path": b"/ingest/stream", "query_string": b"",
            "root_path": "", "scheme": "http", "client": ("test", 1),
            "server": ("test", 80), "app": server.app,
            "headers": [(b"host", b"test"), (b"content-type", b"multipart/form-data; boundary=b"),
                        (b"content-length", str(len(body)).encode())],
        }
        with contextlib.suppress(Exception):     # the broken send propagates; that's the point
            await server.app(scope, receive, send)

        # Asserted INSIDE the loop, deliberately. `asyncio.run` ends by calling
        # `loop.shutdown_asyncgens()`, which closes the `@asynccontextmanager` generator
        # behind `acquire()` and releases the lock as a side effect - so a check after
        # `asyncio.run` returns passes against the leaking implementation too. A real
        # uvicorn process never tears its loop down between requests, so that cleanup does
        # not exist in production and the check has to happen where production would be.
        assert not server._gpu._lock.locked(), "the model stayed locked - the server is wedged"
        assert not server._gpu.contended()

    asyncio.run(drive())


def test_ingest_stream_reports_a_lost_shed_race_in_the_stream(warm, monkeypatch, tmp_path):
    """The door probe is advisory, so the generator's own acquire must degrade cleanly.

    Headers are already out by then, so a lost race cannot become a 503 — it has to be a
    terminal `error` frame rather than a stream that just stops.
    """
    monkeypatch.setattr(server, "PDFS_DIR", tmp_path)
    monkeypatch.setattr(server, "run_ingest",
                        lambda *a, **k: pytest.fail("the build must never start"))

    calls = {"n": 0}
    real_acquire = server._gpu.acquire

    def flaky_acquire(*args, **kwargs):
        """Let the door probe through, then shed the generator's acquire."""
        calls["n"] += 1
        if calls["n"] == 1:
            return real_acquire(*args, **kwargs)

        class _Busy:
            async def __aenter__(self):
                raise server.GpuBusy(30.0)

            async def __aexit__(self, *exc):
                return False

        return _Busy()

    monkeypatch.setattr(server._gpu, "acquire", flaky_acquire)

    resp = warm.post("/ingest/stream",
                     files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")})

    assert resp.status_code == 200                       # headers were already sent
    assert '"phase": "error"' in resp.text                # ...so it says so in the stream
    assert '"retry_after"' in resp.text
    assert not server._gpu._lock.locked()                 # and still no leak


def test_the_server_still_boots_when_the_corpus_check_fails(monkeypatch):
    """Belt and braces on index_health's own guarantee: boot survives a broken check.

    A fresh deployment has no collection until the first ingest, and the boot-time
    integrity check used to raise straight through lifespan - uvicorn refused to start.
    Config errors should fail fast; a diagnostic should not.
    """
    monkeypatch.setattr(server, "validate", lambda: None)
    monkeypatch.setattr(server, "load_model", lambda: None)
    monkeypatch.setattr(server, "get_graph", lambda: None)
    monkeypatch.setattr(server, "ping", lambda: None)
    monkeypatch.setattr(server, "is_loaded", lambda: True)
    monkeypatch.setattr(server, "close_client", lambda: None)

    def boom():
        raise ValueError("Collection pdf_pages not found")

    monkeypatch.setattr(server, "index_health", boom)

    with TestClient(server.app) as client:          # lifespan runs here; must not raise
        assert client.get("/health").status_code == 200
