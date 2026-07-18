"""Ingest CLI: PDF pages -> page PNGSs -> ColPali multivectors -> Qdrant."""

import sys
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

def ingest_pdf(pdf_path: Path, start_id: int, collection_name: str) -> int:
    """Render, embed, and store every page of one PDF, flushing in batches.

    Upserts target `collection_name` explicitly: during a server ingest that is the
    freshly-built physical collection, not the live alias.
    """
    print(f"\nRendering{pdf_path.name} ...")
    pages = pdf_to_images(pdf_path)
    print(f"{len(pages)} pages")

    batch: list = []
    for offset, page in enumerate(pages):
        page_number = offset + 1
        image_path = save_page_image(page, pdf_path.name, page_number)
        multivector = embed_image(page)
        batch.append(build_point(start_id + offset, multivector, pdf_path.name, page_number, image_path))
        print(f"embedded page {page_number}")
        if len(batch) >= UPSERT_BATCH_SIZE:
            upsert_pages(batch, collection_name=collection_name)
            print(f"stored {len(batch)} pages")
            batch = []

    upsert_pages(batch, collection_name=collection_name)  # flush the remainder
    if batch:
        print(f"stored {len(batch)} pages")

    return start_id + len(pages)


def main(pdf_args: list[str]) -> None:
    """Resolve PDFs, then build the Qdrant index atomically.

    Server mode builds a fresh versioned collection and alias-swaps it live only on
    success; an interrupted or failed build drops the partial and leaves the
    previous index serving. Embedded mode wipes and rebuilds in place.
    """
    pdfs = [Path(p) for p in pdf_args] if pdf_args else sorted(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFS found. Put PDF in {PDFS_DIR} or pass a path.")
        sys.exit(1)

    ping()  # fail fast with a clear message before the expensive render/embed loop
    target = begin_ingest()
    next_id = 0

    try:
        for pdf in pdfs:
            if pdf.exists():
                next_id = ingest_pdf(pdf, next_id, collection_name=target)
        finish_ingest(target)  # atomic alias swap (server) / no-op (embedded)
    except BaseException:  # incl. KeyboardInterrupt: drop the partial, keep old index
        abort_ingest(target)
        raise
    finally:
        close_client()

    print(f"\nDone. Indexed {next_id} pages into Qdrant.")

if __name__ == "__main__":
    main(sys.argv[1:])