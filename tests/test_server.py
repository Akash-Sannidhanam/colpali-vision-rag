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
    """A warm server with a reachable Qdrant reports ok."""
    resp = warm.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True, "qdrant": "ok"}


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

    def fake_run_ingest(paths):
        """Record what the endpoint handed the ingest, and report a page count."""
        captured["paths"] = paths
        return 7

    monkeypatch.setattr(server, "PDFS_DIR", tmp_path)                  # don't write into the repo's pdfs/
    monkeypatch.setattr(server, "run_ingest", fake_run_ingest)
    resp = warm.post("/ingest", files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert resp.status_code == 200
    assert resp.json() == {"pdf": "doc.pdf", "indexed_pages": 7}
    assert (tmp_path / "doc.pdf").read_bytes() == b"%PDF-1.4 fake"     # saved under PDFS_DIR
    assert captured["paths"] == [tmp_path / "doc.pdf"]                 # run_ingest got the saved path


def test_ingest_does_not_re_embed_the_rest_of_the_corpus(warm, monkeypatch, tmp_path):
    """Only the uploaded PDF is ingested - adding one document no longer re-embeds the corpus."""
    # An upload used to hand run_ingest every PDF in PDFS_DIR, so adding one document
    # re-rendered and re-embedded the whole corpus through the ~2B model.
    (tmp_path / "already_indexed.pdf").write_bytes(b"%PDF-1.4 old")
    captured = {}
    monkeypatch.setattr(server, "PDFS_DIR", tmp_path)
    monkeypatch.setattr(server, "run_ingest", lambda paths: captured.update(paths=paths) or 2)

    warm.post("/ingest", files={"file": ("new.pdf", b"%PDF-1.4 new", "application/pdf")})

    assert captured["paths"] == [tmp_path / "new.pdf"]                 # only the upload


def test_ingest_rejects_non_pdf(warm):
    """A non-PDF upload is rejected before anything is written."""
    resp = warm.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_ingest_stream_emits_progress_and_done(warm, monkeypatch, tmp_path):
    """The SSE endpoint streams per-page progress and a final done frame."""
    monkeypatch.setattr(server, "PDFS_DIR", tmp_path)

    def fake_run_ingest(paths, progress):
        """Record what the endpoint handed the ingest, and report a page count."""
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
