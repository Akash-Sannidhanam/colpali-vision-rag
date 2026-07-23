"""Turn PDF pages into images- the only parsing this RAG ever does."""
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from src.config import PAGE_IMAGES_DIR, POPPLER_PATH, RENDER_DPI


def pdf_to_images(pdf_path: Path, dpi: int = RENDER_DPI) -> list[Image.Image]:
    """Render every page of a PDF to an RGB PIL image, in page order."""
    poppler = POPPLER_PATH if POPPLER_PATH else None
    # None means "use PATH" at runtime; pdf2image's stub omits None from the type.
    return convert_from_path(str(pdf_path), dpi=dpi, fmt="RGB", poppler_path=poppler)  # type: ignore[arg-type]

def page_image_path(pdf_name: str, page_number: int) -> Path:
    """The deterministic PNG path for one rendered page (no I/O).

    The single source of truth for the `<pdf-stem>_page_<n>.png` naming, so both the
    ingest writer and readers (e.g. the heatmap endpoint resolving a page by pdf+page)
    agree without re-deriving it.
    """
    return PAGE_IMAGES_DIR / f"{Path(pdf_name).stem}_page_{page_number}.png"

def save_page_image(image: Image.Image, pdf_name: str, page_number: int) -> Path:
    """Save one rendered page as a PNG and return its path."""
    PAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = page_image_path(pdf_name, page_number)
    image.save(out_path, format="PNG")
    return out_path