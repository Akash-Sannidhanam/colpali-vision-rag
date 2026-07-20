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

import base64
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import server


def _canned_result(found: bool = True) -> dict:
    """A run_query-shaped RAGState dict: two reranked pages + a meta block with stages."""
    retrieved = [
        {"pdf": "attention.pdf", "page_number": 3, "score": 14.2,
         "image_path": str(server.PAGE_IMAGES_DIR / "attention_page_3.png")},
        {"pdf": "attention.pdf", "page_number": 5, "score": 8.1,
         "image_path": str(server.PAGE_IMAGES_DIR / "attention_page_5.png")},
    ]
    if found:
        citation = {"answer": "42", "found": True, "source_page": 1, "box": [140, 300, 660, 700]}
        crop = str(server.PAGE_IMAGES_DIR / "crops" / "attention_page_3_crop.png")
        annotated = str(server.PAGE_IMAGES_DIR / "crops" / "attention_page_3_annotated.png")
    else:
        citation = {"answer": "Couldn't find it.", "found": False, "source_page": 0, "box": []}
        crop = annotated = None
    return {
        "question": "what?",
        "retrieved": retrieved,
        "answer": citation["answer"],
        "citation": citation,
        "crop_path": crop,
        "annotated_path": annotated,
        "meta": {
            "request_id": "abc123", "latency_ms": 3400.0,
            "prompt_tokens": 9000, "output_tokens": 2000, "total_tokens": 11000,
            "est_cost_usd": 0.015, "gemini_calls": 2,
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
    monkeypatch.setattr(server, "validate", lambda: None)
    monkeypatch.setattr(server, "load_model", lambda: None)
    monkeypatch.setattr(server, "get_graph", lambda: None)
    monkeypatch.setattr(server, "ping", lambda: None)
    monkeypatch.setattr(server, "is_loaded", lambda: True)
    monkeypatch.setattr(server, "close_client", lambda: None)
    with TestClient(server.app) as client:
        yield client


# --- /health ---

def test_health_ok(warm):
    resp = warm.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True, "qdrant": "ok"}


def test_health_degraded_when_qdrant_unreachable(warm, monkeypatch):
    def down():
        raise RuntimeError("Cannot reach Qdrant")

    monkeypatch.setattr(server, "ping", down)          # after startup, so boot still succeeded
    resp = warm.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert "Cannot reach Qdrant" in body["qdrant"]


# --- /corpus ---

def test_corpus_lists_documents_and_total(warm, monkeypatch):
    monkeypatch.setattr(server, "list_documents",
                        lambda: [{"pdf": "attention.pdf", "page_count": 15},
                                 {"pdf": "colpali.pdf", "page_count": 26}])
    resp = warm.get("/corpus")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_pages"] == 41
    assert body["qdrant"] == "ok"
    assert [d["pdf"] for d in body["documents"]] == ["attention.pdf", "colpali.pdf"]


# --- /query ---

def test_query_happy_path_shape(warm, monkeypatch):
    monkeypatch.setattr(server, "run_query", lambda q: _canned_result(found=True))
    resp = warm.post("/query", json={"question": "what was the Q4 revenue?"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"] == "42"
    cit = body["citation"]
    assert cit["found"] is True and cit["source_page"] == 1
    assert cit["box"] == [140, 300, 660, 700]
    assert cit["pdf"] == "attention.pdf" and cit["page_number"] == 3   # enriched from pages[0]

    assert len(body["pages"]) == 2
    assert body["pages"][0]["image"]["url"].endswith("/images/attention_page_3.png")
    assert body["pages"][0]["image"]["data_uri"] is None               # url mode by default

    assert body["crop"]["url"].endswith("attention_page_3_crop.png")
    assert body["annotated"]["url"].endswith("attention_page_3_annotated.png")

    meta = body["meta"]
    assert meta["request_id"] == "abc123"
    assert meta["total_tokens"] == 11000 and meta["est_cost_usd"] == 0.015
    assert meta["retrieve_k"] == server.RETRIEVE_K
    assert [s["node"] for s in meta["stages"]] == ["retrieve", "rerank", "answer", "highlight"]


def test_query_not_found_has_no_crop(warm, monkeypatch):
    monkeypatch.setattr(server, "run_query", lambda q: _canned_result(found=False))
    resp = warm.post("/query", json={"question": "unanswerable?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citation"]["found"] is False and body["citation"]["source_page"] == 0
    assert body["citation"]["pdf"] is None
    assert body["crop"] is None and body["annotated"] is None
    assert len(body["pages"]) == 2                                     # candidates still surfaced


def test_query_inline_populates_data_uri(warm, monkeypatch):
    monkeypatch.setattr(server, "run_query", lambda q: _canned_result(found=True))
    monkeypatch.setattr(server, "_encode_data_uri", lambda p: "data:image/png;base64,STUB")
    resp = warm.post("/query?inline=true", json={"question": "q"})
    body = resp.json()
    assert body["pages"][0]["image"]["data_uri"] == "data:image/png;base64,STUB"
    assert body["pages"][0]["image"]["url"] is not None                # url stays populated too
    assert body["crop"]["data_uri"] == "data:image/png;base64,STUB"


def test_query_empty_question_is_422(warm):
    assert warm.post("/query", json={"question": ""}).status_code == 422
    assert warm.post("/query", json={}).status_code == 422


# --- /ingest ---

def test_ingest_happy_path(warm, monkeypatch, tmp_path):
    captured = {}

    def fake_run_ingest(paths):
        captured["paths"] = paths
        return 7

    monkeypatch.setattr(server, "PDFS_DIR", tmp_path)                  # don't write into the repo's pdfs/
    monkeypatch.setattr(server, "run_ingest", fake_run_ingest)
    resp = warm.post("/ingest", files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert resp.status_code == 200
    assert resp.json() == {"pdf": "doc.pdf", "indexed_pages": 7}
    assert (tmp_path / "doc.pdf").read_bytes() == b"%PDF-1.4 fake"     # saved under PDFS_DIR
    assert captured["paths"] == [tmp_path / "doc.pdf"]                 # run_ingest got the saved path


def test_ingest_rejects_non_pdf(warm):
    resp = warm.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_ingest_rejects_oversize(warm, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_MB", 0)                    # anything non-empty is too big
    resp = warm.post("/ingest", files={"file": ("big.pdf", b"%PDF-1.4 xxxxxxxx", "application/pdf")})
    assert resp.status_code == 413


# --- pure image helpers (no server) ---

def test_to_url_maps_page_images_path(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "PAGE_IMAGES_DIR", tmp_path)
    fs = tmp_path / "crops" / "sales_report_page_1_crop.png"
    fs.parent.mkdir(parents=True)
    fs.write_bytes(b"x")
    url = server._to_url(SimpleNamespace(base_url="http://testserver/"), str(fs))
    assert url == "http://testserver/images/crops/sales_report_page_1_crop.png"


def test_encode_data_uri_roundtrips(tmp_path):
    fs = tmp_path / "p.png"
    payload = b"\x89PNG\r\n fake bytes"
    fs.write_bytes(payload)
    uri = server._encode_data_uri(str(fs))
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == payload
