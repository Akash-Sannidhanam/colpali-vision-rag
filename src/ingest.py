"""Ingest CLI: PDF pages -> page PNGSs -> ColPali multivectors -> Qdrant."""

import sys
from pathlib import Path 

from src.config import PDFS_DIR, UPSERT_BATCH_SIZE
from src.embedder import embed_image
from src.pdf_render import pdf_to_images, save_page_image
from src.vector_store import build_point, close_client, ensure_collection, upsert_pages

def ingest_pdf(pdf_path: Path, start_id: int) -> int:
    """Render, embed, and store every page of one PDF, flushing in batches."""
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
            upsert_pages(batch)
            print(f"stored {len(batch)} pages")
            batch = []

    upsert_pages(batch)  # flush the remainder
    if batch:
        print(f"stored {len(batch)} pages")

    return start_id + len(pages)


def main(pdf_args: list[str]) -> None:
    """Resolve PDFs, then build the Qdrant collection fresh."""
    pdfs = [Path(p) for p in pdf_args] if pdf_args else sorted(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFS found. Put PDF in {PDFS_DIR} or pass a path.")
        sys.exit(1)
        
    ensure_collection(reset=True)
    next_id = 0

    try:
        for pdf in pdfs:
            if pdf.exists():
                next_id = ingest_pdf(pdf, next_id)
    finally:
        close_client()

    print(f"\nDone. Indexed {next_id} pages into Qdrant.")

if __name__ == "__main__":
    main(sys.argv[1:])