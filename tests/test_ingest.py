"""Tests for ingest progress events and the incremental sync logic (src.ingest).

Stubs the render/embed/store boundaries as imported into `ingest`, so no poppler,
model, or Qdrant is touched - the suite verifies run_ingest emits one progress event
per render/embed/store step (the data the server's SSE endpoint streams), that the
default (no callback) path still prints for the CLI, and that the sync path re-embeds
exactly the documents whose bytes or embedding config changed.
"""

import pytest

from src import ingest


def _stub_pipeline(monkeypatch, pages_per_pdf: int, indexed: dict | None = None) -> list[str]:
    """Stub every I/O boundary; return the list that records embedded PDF names."""
    embedded: list[str] = []
    monkeypatch.setattr(ingest, "ping", lambda: None)
    monkeypatch.setattr(ingest, "begin_ingest", lambda: "pdf_pages_1")
    monkeypatch.setattr(ingest, "finish_ingest", lambda target: None)
    monkeypatch.setattr(ingest, "abort_ingest", lambda target: None)
    monkeypatch.setattr(ingest, "live_collection", lambda: "pdf_pages")
    monkeypatch.setattr(ingest, "document_index", lambda: dict(indexed or {}))
    monkeypatch.setattr(ingest, "delete_document", lambda name: 0)
    monkeypatch.setattr(ingest, "pdf_to_images", lambda path: [object()] * pages_per_pdf)
    monkeypatch.setattr(ingest, "save_page_image",
                        lambda page, name, n: embedded.append(name) or f"{name}_p{n}.png")
    # Yields in batches of 3 rather than one big batch, so the page numbering ingest derives
    # from each batch's start index (`start + i + 1`) is actually exercised across batches.
    def _iter_embedded(pages, batch_size=None):
        for start in range(0, len(pages), 3):
            yield start, [[[0.0] * 128] for _ in pages[start:start + 3]]

    monkeypatch.setattr(ingest, "iter_embedded", _iter_embedded)
    monkeypatch.setattr(ingest, "build_point", lambda *a: {"point": a})
    monkeypatch.setattr(ingest, "upsert_pages", lambda batch, collection_name: None)
    return embedded


def _pdf(tmp_path, name: str = "doc.pdf", body: bytes = b"%PDF-1.4"):
    """Write a stub PDF into tmp_path and return its path."""
    path = tmp_path / name
    path.write_bytes(body)
    return path


# --- progress events ---

def test_run_ingest_emits_progress_per_step(monkeypatch, tmp_path):
    """One render/pages/embed/stored event per step - the data the SSE endpoint streams."""
    _stub_pipeline(monkeypatch, pages_per_pdf=3)
    pdf = _pdf(tmp_path)

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
    """With no callback, the CLI's inline print path still runs unchanged."""
    _stub_pipeline(monkeypatch, pages_per_pdf=1)
    ingest.run_ingest([_pdf(tmp_path)])   # no callback -> the CLI's inline print path

    out = capsys.readouterr().out
    assert "Rendering doc.pdf" in out and "embedded page 1" in out


def test_the_whole_document_goes_to_one_embed_generator(monkeypatch, tmp_path):
    """Ingest must not pre-chunk pages into EMBED_BATCH_SIZE windows.

    `iter_embedded` halves its batch on an out-of-memory error and keeps the smaller size
    for the life of the generator. Calling it once per window throws that away - each call
    restarts at EMBED_BATCH_SIZE, so a memory-tight box re-pays the failed forward pass on
    every window instead of once per document. This test is the guard: one call, all pages.
    """
    _stub_pipeline(monkeypatch, pages_per_pdf=10)
    calls: list[int] = []

    def _recording_iter(pages, batch_size=None):
        calls.append(len(pages))
        for start in range(0, len(pages), 3):
            yield start, [[[0.0] * 128] for _ in pages[start:start + 3]]

    monkeypatch.setattr(ingest, "iter_embedded", _recording_iter)
    pages: list[int] = []
    ingest.run_ingest([_pdf(tmp_path)],
                      progress=lambda e: e["phase"] == "embed" and pages.append(e["page"]))

    assert calls == [10]                    # one generator for the document, not one per window
    assert pages == list(range(1, 11))      # and still one in-order embed event per page


# --- incremental sync: what gets re-embedded ---

def test_sync_skips_a_document_whose_fingerprint_still_matches(monkeypatch, tmp_path):
    """Matching content hash + embed version means zero pages embedded and no model call."""
    pdf = _pdf(tmp_path)
    indexed = {"doc.pdf": {"page_count": 3,
                           "content_hash": ingest._fingerprint(pdf),
                           "embed_version": ingest.EMBED_VERSION}}
    embedded = _stub_pipeline(monkeypatch, pages_per_pdf=3, indexed=indexed)

    events: list[dict] = []
    total = ingest.run_ingest([pdf], progress=events.append)

    assert total == 0                                    # nothing embedded this run
    assert embedded == []                                # the model was never touched
    assert events == [{"phase": "skip", "pdf": "doc.pdf", "total": 3}]


def test_sync_re_embeds_when_the_content_hash_changes(monkeypatch, tmp_path):
    """Edited PDF bytes invalidate the stored vectors and force a re-embed."""
    pdf = _pdf(tmp_path)
    indexed = {"doc.pdf": {"page_count": 3, "content_hash": "stale",
                           "embed_version": ingest.EMBED_VERSION}}
    embedded = _stub_pipeline(monkeypatch, pages_per_pdf=3, indexed=indexed)

    assert ingest.run_ingest([pdf], progress=lambda e: None) == 3
    assert embedded == ["doc.pdf"] * 3


def test_sync_re_embeds_when_the_embed_version_changes(monkeypatch, tmp_path):
    """A rebuild writes hashes too, or the next sync would re-embed the whole corpus."""
    # Same PDF bytes, different model/DPI: a content hash alone would wrongly skip this.
    """A model/DPI change re-embeds even though the bytes are identical."""
    pdf = _pdf(tmp_path)
    indexed = {"doc.pdf": {"page_count": 3,
                           "content_hash": ingest._fingerprint(pdf),
                           "embed_version": "some-other-model@72"}}
    embedded = _stub_pipeline(monkeypatch, pages_per_pdf=3, indexed=indexed)

    assert ingest.run_ingest([pdf], progress=lambda e: None) == 3
    assert embedded == ["doc.pdf"] * 3


def test_sync_deletes_before_re_embedding_a_changed_document(monkeypatch, tmp_path):
    # Point ids are stable per (pdf, page), so upserting a shorter revision would
    # overwrite pages 1..n and strand the rest. The delete is what prevents that.
    """The delete precedes the render, so a shorter revision can't strand tail pages."""
    pdf = _pdf(tmp_path)
    indexed = {"doc.pdf": {"page_count": 9, "content_hash": "stale",
                           "embed_version": ingest.EMBED_VERSION}}
    _stub_pipeline(monkeypatch, pages_per_pdf=2, indexed=indexed)
    order: list[str] = []
    monkeypatch.setattr(ingest, "delete_document", lambda name: order.append(f"delete:{name}"))
    monkeypatch.setattr(ingest, "pdf_to_images",
                        lambda path: order.append("render") or [object()] * 2)

    ingest.run_ingest([pdf], progress=lambda e: None)

    assert order == ["delete:doc.pdf", "render"]


def test_sync_does_not_delete_a_document_it_has_never_seen(monkeypatch, tmp_path):
    """A brand-new document is upserted directly, with no pointless delete first."""
    _stub_pipeline(monkeypatch, pages_per_pdf=1, indexed={})
    deletes: list[str] = []
    monkeypatch.setattr(ingest, "delete_document", lambda name: deletes.append(name))

    ingest.run_ingest([_pdf(tmp_path)], progress=lambda e: None)

    assert deletes == []


def test_sync_writes_into_the_live_collection_not_a_new_version(monkeypatch, tmp_path):
    """Sync upserts into the live alias and never starts a versioned rebuild."""
    _stub_pipeline(monkeypatch, pages_per_pdf=1)
    targets: list[str] = []
    monkeypatch.setattr(ingest, "upsert_pages",
                        lambda batch, collection_name: targets.append(collection_name))
    monkeypatch.setattr(ingest, "begin_ingest",
                        lambda: pytest.fail("sync must not start a versioned rebuild"))

    ingest.run_ingest([_pdf(tmp_path)], progress=lambda e: None)

    assert set(targets) == {"pdf_pages"}


def test_sync_embeds_only_the_changed_document_in_a_mixed_corpus(monkeypatch, tmp_path):
    """Only the stale document is re-embedded; the current one is left alone."""
    fresh, stale = _pdf(tmp_path, "fresh.pdf"), _pdf(tmp_path, "stale.pdf", b"%PDF-x")
    indexed = {"fresh.pdf": {"page_count": 2,
                             "content_hash": ingest._fingerprint(fresh),
                             "embed_version": ingest.EMBED_VERSION},
               "stale.pdf": {"page_count": 2, "content_hash": "old",
                             "embed_version": ingest.EMBED_VERSION}}
    embedded = _stub_pipeline(monkeypatch, pages_per_pdf=2, indexed=indexed)

    assert ingest.run_ingest([fresh, stale], progress=lambda e: None) == 2
    assert set(embedded) == {"stale.pdf"}


def test_sync_never_prunes_documents_missing_from_disk(monkeypatch, tmp_path):
    # A document that is indexed but no longer passed in is left alone: removal is
    # always explicit, never inferred from a file's absence.
    """An indexed document absent from the run is left alone, never auto-deleted."""
    indexed = {"gone.pdf": {"page_count": 4, "content_hash": "h",
                            "embed_version": ingest.EMBED_VERSION}}
    _stub_pipeline(monkeypatch, pages_per_pdf=1, indexed=indexed)
    deletes: list[str] = []
    monkeypatch.setattr(ingest, "delete_document", lambda name: deletes.append(name))

    ingest.run_ingest([_pdf(tmp_path)], progress=lambda e: None)

    assert deletes == []


# --- rebuild: the atomic escape hatch ---

def test_rebuild_takes_the_atomic_path_and_ignores_fingerprints(monkeypatch, tmp_path):
    """--rebuild re-embeds everything and publishes via the alias swap."""
    pdf = _pdf(tmp_path)
    indexed = {"doc.pdf": {"page_count": 3,
                           "content_hash": ingest._fingerprint(pdf),
                           "embed_version": ingest.EMBED_VERSION}}
    embedded = _stub_pipeline(monkeypatch, pages_per_pdf=3, indexed=indexed)
    finished: list[str] = []
    monkeypatch.setattr(ingest, "finish_ingest", lambda target: finished.append(target))

    total = ingest.run_ingest([pdf], progress=lambda e: None, rebuild=True)

    assert total == 3
    assert embedded == ["doc.pdf"] * 3      # rebuilt even though the fingerprint matched
    assert finished == ["pdf_pages_1"]      # published via the alias swap


def test_rebuild_aborts_the_partial_on_failure(monkeypatch, tmp_path):
    """A mid-build failure drops the partial so the previous index keeps serving."""
    _stub_pipeline(monkeypatch, pages_per_pdf=1)
    aborted: list[str] = []
    monkeypatch.setattr(ingest, "abort_ingest", lambda target: aborted.append(target))
    monkeypatch.setattr(ingest, "iter_embedded", lambda pages: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        ingest.run_ingest([_pdf(tmp_path)], progress=lambda e: None, rebuild=True)

    assert aborted == ["pdf_pages_1"]       # the previous index keeps serving


def test_rebuild_stores_fingerprints_so_the_next_sync_can_skip(monkeypatch, tmp_path):
    """A rebuild writes fingerprints too - empty ones would make the very next sync
    re-embed the whole corpus again."""
    pdf = _pdf(tmp_path)
    _stub_pipeline(monkeypatch, pages_per_pdf=1)
    points: list[tuple] = []
    monkeypatch.setattr(ingest, "build_point", lambda *a: points.append(a) or {"p": a})

    ingest.run_ingest([pdf], progress=lambda e: None, rebuild=True)

    _multivector, name, page, _image, content_hash, embed_version = points[0]
    assert name == "doc.pdf" and page == 1
    assert content_hash == ingest._fingerprint(pdf)
    assert embed_version == ingest.EMBED_VERSION


# --- CLI wrapper ---

def test_cli_reports_skipped_documents(monkeypatch, tmp_path, capsys):
    """The CLI summary reports how many documents were already up to date."""
    pdf = _pdf(tmp_path)
    indexed = {"doc.pdf": {"page_count": 3,
                           "content_hash": ingest._fingerprint(pdf),
                           "embed_version": ingest.EMBED_VERSION}}
    _stub_pipeline(monkeypatch, pages_per_pdf=3, indexed=indexed)
    monkeypatch.setattr(ingest, "close_client", lambda: None)

    ingest.main([str(pdf)])

    out = capsys.readouterr().out
    assert "Embedded 0 pages, 1 document already up to date." in out


def test_cli_rebuild_flag_is_not_treated_as_a_path(monkeypatch, tmp_path, capsys):
    """--rebuild sets the flag and is stripped from the PDF path list."""
    _stub_pipeline(monkeypatch, pages_per_pdf=2)
    monkeypatch.setattr(ingest, "close_client", lambda: None)
    seen: dict = {}
    monkeypatch.setattr(ingest, "run_ingest",
                        lambda pdfs, progress, rebuild: seen.update(pdfs=pdfs, rebuild=rebuild) or 2)

    ingest.main([str(_pdf(tmp_path)), "--rebuild"])

    assert seen["rebuild"] is True
    assert [p.name for p in seen["pdfs"]] == ["doc.pdf"]     # the flag is not a PDF path
    assert "Rebuilt the index with 2 pages." in capsys.readouterr().out


# --- the store worker behind the forward pass ---

def test_a_failing_store_worker_surfaces_instead_of_hanging(monkeypatch, tmp_path):
    """A dead worker must not leave the embed loop blocked on a full queue.

    An ingest that hangs is strictly worse than one that crashes: `document_index`
    fingerprints a document from its first indexed page, so a killed run strands a truncated
    document that the next sync considers current and skips. The queue is bounded, so
    without the drain-to-sentinel in `_StoreWorker._run` this deadlocks rather than fails.
    """
    _stub_pipeline(monkeypatch, pages_per_pdf=40)

    def _explode(batch, collection_name):
        raise RuntimeError("qdrant is down")

    monkeypatch.setattr(ingest, "upsert_pages", _explode)

    with pytest.raises(RuntimeError, match="qdrant is down"):
        ingest.run_ingest([_pdf(tmp_path)])


def test_the_bodys_own_failure_is_not_masked_by_the_worker(monkeypatch, tmp_path):
    """An OOM in the embed loop is the real story; shutting the worker down must not hide it."""
    _stub_pipeline(monkeypatch, pages_per_pdf=6)

    def _boom(pages, batch_size=None):
        yield 0, [[[0.0] * 128]]
        raise MemoryError("out of memory")

    monkeypatch.setattr(ingest, "iter_embedded", _boom)

    with pytest.raises(MemoryError, match="out of memory"):
        ingest.run_ingest([_pdf(tmp_path)])


def test_pages_are_stored_in_page_order(monkeypatch, tmp_path):
    """One worker draining a FIFO queue, so moving the upsert off-thread cannot reorder pages."""
    _stub_pipeline(monkeypatch, pages_per_pdf=10)
    stored: list[int] = []
    monkeypatch.setattr(ingest, "build_point",
                        lambda mv, name, page, path, ch, ev: stored.append(page) or {"p": page})

    ingest.run_ingest([_pdf(tmp_path)])

    assert stored == list(range(1, 11))


def test_every_page_reaches_qdrant_exactly_once(monkeypatch, tmp_path):
    """The remainder past the last full UPSERT_BATCH_SIZE flush must still be stored."""
    _stub_pipeline(monkeypatch, pages_per_pdf=10)   # 10 pages, flushes at 8, remainder 2
    upserted: list[int] = []
    monkeypatch.setattr(ingest, "upsert_pages",
                        lambda batch, collection_name: upserted.append(len(batch)))

    ingest.run_ingest([_pdf(tmp_path)])

    assert sum(upserted) == 10
