"""Ingest CLI: PDF pages -> page PNGSs -> ColPali multivectors -> Qdrant."""

import sys
from collections.abc import Callable
from pathlib import Path

from src.config import PDFS_DIR, UPSERT_BATCH_SIZE
from src.embedder import embed_image
from src.pdf_render import pdf_to_images, save_page_image
from src.vector_store import (
    abort_ingest,
    begin_ingest,
    build_point,
    close_client,
    finish_ingest,
    ping,
    upsert_pages,
)

# A progress event is a small dict: {"phase": render|pages|embed|stored, ...}. The
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


def ingest_pdf(pdf_path: Path, start_id: int, collection_name: str, progress: Progress) -> int:
    """Render, embed, and store every page of one PDF, flushing in batches.

    Upserts target `collection_name` explicitly: during a server ingest that is the
    freshly-built physical collection, not the live alias. Emits a progress event at
    each render/embed/store step so a caller can stream per-page updates.
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
        batch.append(build_point(start_id + offset, multivector, name, page_number, image_path))
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

    return start_id + total


def run_ingest(pdfs: list[Path], progress: Progress | None = None) -> int:
    """Build the Qdrant index from `pdfs` atomically and return the page count.

    The client-teardown-free core seam (mirrors `main.run_query` vs `run`): a warm
    server calls this and keeps the shared Qdrant client open, while the CLI `main()`
    wrapper owns closing it. Server mode builds a fresh versioned collection and
    alias-swaps it live only on success; an interrupted or failed build drops the
    partial and leaves the previous index serving. Embedded mode wipes and rebuilds
    in place.

    `progress` receives one event dict per render/embed/store step; it defaults to
    printing the same lines the CLI always has, so the CLI is unchanged.
    """
    emit = progress or _print_progress
    ping()  # fail fast with a clear message before the expensive render/embed loop
    target = begin_ingest()
    next_id = 0
    try:
        for pdf in pdfs:
            if pdf.exists():
                next_id = ingest_pdf(pdf, next_id, collection_name=target, progress=emit)
        finish_ingest(target)  # atomic alias swap (server) / no-op (embedded)
    except BaseException:  # incl. KeyboardInterrupt: drop the partial, keep old index
        abort_ingest(target)
        raise
    return next_id


def main(pdf_args: list[str]) -> None:
    """Resolve PDFs and build the index (CLI wrapper; owns Qdrant client teardown)."""
    pdfs = [Path(p) for p in pdf_args] if pdf_args else sorted(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFS found. Put PDF in {PDFS_DIR} or pass a path.")
        sys.exit(1)

    try:
        next_id = run_ingest(pdfs)
    finally:
        close_client()

    print(f"\nDone. Indexed {next_id} pages into Qdrant.")

if __name__ == "__main__":
    main(sys.argv[1:])