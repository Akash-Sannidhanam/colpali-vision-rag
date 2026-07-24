"""Ingest CLI: PDF pages -> page PNGSs -> ColPali multivectors -> Qdrant.

Two modes, both driven from `run_ingest`:

- **Sync** (default) - upsert into the live collection, embedding only the documents
  whose bytes or embedding config changed. Adding a document costs its own pages, and
  re-running over an unchanged corpus is a no-op.
- **Rebuild** (`rebuild=True`, `--rebuild`) - the original atomic wholesale build:
  everything is re-embedded into a fresh collection that is published only on success.
  The escape hatch for a genuine wipe, and the way to reclaim space after documents
  have been deleted.

Sync never prunes: a document that is indexed but no longer on disk is left alone.
Removal is always explicit (`vector_store.delete_document`, `DELETE /corpus/{pdf}`) or
wholesale (`--rebuild`), never inferred from a file's absence.
"""

import hashlib
import sys
from collections.abc import Callable
from pathlib import Path

from src.config import EMBED_VERSION, PDFS_DIR, UPSERT_BATCH_SIZE
from src.embedder import embed_image
from src.pdf_render import pdf_to_images, save_page_image
from src.vector_store import (
    abort_ingest,
    begin_ingest,
    build_point,
    close_client,
    delete_document,
    document_index,
    finish_ingest,
    live_collection,
    ping,
    upsert_pages,
)

# A progress event is a small dict: {"phase": render|pages|embed|stored|skip, ...}. The
# callback is invoked from whatever thread runs the ingest, so a consumer that touches
# an event loop (the server's SSE endpoint) must hop back to it.
Progress = Callable[[dict], None]


def _print_progress(evt: dict) -> None:
    """Default progress sink: the human-readable lines the CLI used to print inline."""
    phase = evt["phase"]
    if phase == "render":
        print(f"\nRendering {evt['pdf']} ...")
    elif phase == "pages":
        print(f"{evt['total']} pages")
    elif phase == "embed":
        print(f"embedded page {evt['page']}")
    elif phase == "stored":
        print(f"stored {evt['count']} pages")
    elif phase == "skip":
        print(f"\n{evt['pdf']} unchanged ({evt['total']} pages) - skipping")


def _fingerprint(pdf_path: Path) -> str:
    """A content hash of the PDF's bytes, read in chunks so a big file isn't slurped."""
    digest = hashlib.sha256()
    with pdf_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_pdf(
    pdf_path: Path, collection_name: str, progress: Progress, content_hash: str = "",
) -> int:
    """Render, embed, and store every page of one PDF, flushing in batches; return the
    page count.

    Upserts target `collection_name` explicitly: during a rebuild that is the freshly-built
    physical collection, during a sync it is the live one. Point ids are derived from
    (pdf name, page number) inside `build_point`, so a re-ingest overwrites pages in place
    rather than duplicating them. Emits a progress event at each render/embed/store step so
    a caller can stream per-page updates.
    """
    name = pdf_path.name
    progress({"phase": "render", "pdf": name})
    pages = pdf_to_images(pdf_path)
    total = len(pages)
    progress({"phase": "pages", "pdf": name, "total": total})

    batch: list = []
    stored = 0
    for offset, page in enumerate(pages):
        page_number = offset + 1
        image_path = save_page_image(page, name, page_number)
        multivector = embed_image(page)
        batch.append(build_point(
            multivector, name, page_number, image_path, content_hash, EMBED_VERSION,
        ))
        progress({"phase": "embed", "pdf": name, "page": page_number, "total": total})
        if len(batch) >= UPSERT_BATCH_SIZE:
            upsert_pages(batch, collection_name=collection_name)
            stored += len(batch)
            progress({"phase": "stored", "pdf": name, "count": stored, "total": total})
            batch = []

    upsert_pages(batch, collection_name=collection_name)  # flush the remainder
    if batch:
        stored += len(batch)
        progress({"phase": "stored", "pdf": name, "count": stored, "total": total})

    return total


def _rebuild(pdfs: list[Path], emit: Progress) -> int:
    """Atomic wholesale build: everything into a fresh collection, published on success."""
    target = begin_ingest()
    total = 0
    try:
        for pdf in pdfs:
            if pdf.exists():
                total += ingest_pdf(pdf, target, emit, _fingerprint(pdf))
        finish_ingest(target)  # atomic alias swap (server) / no-op (embedded)
    except BaseException:  # incl. KeyboardInterrupt: drop the partial, keep old index
        abort_ingest(target)
        raise
    return total


def _sync(pdfs: list[Path], emit: Progress) -> int:
    """Embed only what changed, upserting into the live collection.

    A document is current when both its content hash and the embedding config that
    produced it still match, in which case it is skipped outright. Otherwise its pages
    are deleted first and re-embedded: point ids are stable per (pdf, page), so an
    upsert alone would overwrite pages 1..n but strand pages n+1.. of a longer previous
    revision.

    No rollback: existing documents are never touched, so an interrupted run leaves the
    rest of the index serving and the offending document is completed on the next pass.
    """
    target = live_collection()
    indexed = document_index()
    total = 0
    for pdf in pdfs:
        if not pdf.exists():
            continue
        name = pdf.name
        fingerprint = _fingerprint(pdf)
        current = indexed.get(name)
        if (current
                and current["content_hash"] == fingerprint
                and current["embed_version"] == EMBED_VERSION):
            emit({"phase": "skip", "pdf": name, "total": current["page_count"]})
            continue
        if current:
            delete_document(name)
        total += ingest_pdf(pdf, target, emit, fingerprint)
    return total


def run_ingest(pdfs: list[Path], progress: Progress | None = None, *, rebuild: bool = False) -> int:
    """Index `pdfs` and return the number of pages embedded *this run*.

    The client-teardown-free core seam (mirrors `main.run_query` vs `run`): a warm
    server calls this and keeps the shared Qdrant client open, while the CLI `main()`
    wrapper owns closing it.

    Defaults to the incremental sync; `rebuild=True` takes the atomic wholesale path,
    where an interrupted or failed build drops the partial and leaves the previous index
    serving. A skipped (unchanged) document contributes 0 to the return value, so a
    no-op re-run returns 0.

    `progress` receives one event dict per render/embed/store/skip step; it defaults to
    printing the same lines the CLI always has, so the CLI is unchanged.
    """
    emit = progress or _print_progress
    ping()  # fail fast with a clear message before the expensive render/embed loop
    return _rebuild(pdfs, emit) if rebuild else _sync(pdfs, emit)


def main(argv: list[str]) -> None:
    """Resolve PDFs and index them (CLI wrapper; owns Qdrant client teardown)."""
    rebuild = "--rebuild" in argv
    pdf_args = [a for a in argv if a != "--rebuild"]
    pdfs = [Path(p) for p in pdf_args] if pdf_args else sorted(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFS found. Put PDF in {PDFS_DIR} or pass a path.")
        sys.exit(1)

    skipped = 0

    def progress(evt: dict) -> None:
        """The default printer, plus a tally for the closing summary."""
        nonlocal skipped
        if evt["phase"] == "skip":
            skipped += 1
        _print_progress(evt)

    try:
        indexed = run_ingest(pdfs, progress, rebuild=rebuild)
    finally:
        close_client()

    if rebuild:
        print(f"\nDone. Rebuilt the index with {indexed} pages.")
    else:
        tail = f", {skipped} document{'' if skipped == 1 else 's'} already up to date" if skipped else ""
        print(f"\nDone. Embedded {indexed} pages{tail}.")

if __name__ == "__main__":
    main(sys.argv[1:])
