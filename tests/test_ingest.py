"""Tests for ingest progress events (src.ingest).

Stubs the render/embed/store boundaries as imported into `ingest`, so no poppler,
model, or Qdrant is touched - the suite verifies run_ingest emits one progress event
per render/embed/store step (the data the server's SSE endpoint streams) and that the
default (no callback) path still prints for the CLI.
"""

from src import ingest


def _stub_pipeline(monkeypatch, pages_per_pdf: int) -> None:
    monkeypatch.setattr(ingest, "ping", lambda: None)
    monkeypatch.setattr(ingest, "begin_ingest", lambda: "pdf_pages_1")
    monkeypatch.setattr(ingest, "finish_ingest", lambda target: None)
    monkeypatch.setattr(ingest, "abort_ingest", lambda target: None)
    monkeypatch.setattr(ingest, "pdf_to_images", lambda path: [object()] * pages_per_pdf)
    monkeypatch.setattr(ingest, "save_page_image", lambda page, name, n: f"{name}_p{n}.png")
    monkeypatch.setattr(ingest, "embed_image", lambda page: [[0.0] * 128])
    monkeypatch.setattr(ingest, "build_point", lambda *a: {"id": a[0]})
    monkeypatch.setattr(ingest, "upsert_pages", lambda batch, collection_name: None)


def test_run_ingest_emits_progress_per_step(monkeypatch, tmp_path):
    _stub_pipeline(monkeypatch, pages_per_pdf=3)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    events: list[dict] = []
    total = ingest.run_ingest([pdf], progress=events.append)

    assert total == 3
    assert events[0] == {"phase": "render", "pdf": "doc.pdf"}
    assert events[1] == {"phase": "pages", "pdf": "doc.pdf", "total": 3}
    embeds = [e for e in events if e["phase"] == "embed"]
    assert [e["page"] for e in embeds] == [1, 2, 3]
    assert all(e["total"] == 3 for e in embeds)
    assert any(e["phase"] == "stored" for e in events)


def test_run_ingest_defaults_to_printing(monkeypatch, tmp_path, capsys):
    _stub_pipeline(monkeypatch, pages_per_pdf=1)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    ingest.run_ingest([pdf])   # no callback -> the CLI's inline print path

    out = capsys.readouterr().out
    assert "Rendering doc.pdf" in out and "embedded page 1" in out
