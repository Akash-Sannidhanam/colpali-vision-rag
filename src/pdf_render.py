"""Turn PDF pages into images- the only parsing this RAG ever does."""
import re
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from src.config import CROPS_DIR, PAGE_IMAGES_DIR, POPPLER_PATH, RENDER_DPI


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


def _matching(directory: Path, pattern: re.Pattern[str]) -> list[Path]:
    """Files directly in `directory` whose *name* fully matches `pattern` (missing dir -> [])."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and pattern.fullmatch(p.name))


def page_images_for(pdf_name: str) -> list[Path]:
    """Every rendered page PNG belonging to one PDF (powers document deletion).

    Matched with an anchored regex rather than a `<stem>_page_*` glob on purpose: a
    document named `report_page_1.pdf` renders to `report_page_1_page_1.png`, which a
    loose glob for `report.pdf` would happily sweep up. Deleting the wrong page image
    is unrecoverable, so the page number is required to be the final component.
    """
    stem = re.escape(Path(pdf_name).stem)
    return _matching(PAGE_IMAGES_DIR, re.compile(rf"{stem}_page_\d+\.png"))


# `<stem>_page_<n>.png`, with the page number required to be the final component. The
# stem group is greedy, so `report_page_1_page_2.png` parses as stem `report_page_1`
# page 2 - the same disambiguation `page_images_for`'s per-stem anchored regex makes,
# which is what lets `page_image_numbers` replace N of those calls with one directory scan.
_PAGE_IMAGE_RE = re.compile(r"(?P<stem>.+)_page_(?P<page>\d+)\.png")


def page_image_numbers() -> dict[str, set[int]]:
    """Which page *numbers* are rendered per PDF stem, from one scan of PAGE_IMAGES_DIR.

    The bulk form of `page_images_for`, for callers that need this for every indexed
    document at once (`vector_store.index_health`, `ingest._sync`). Calling
    `page_images_for` per document would re-list a directory holding one file per page
    in the whole corpus, once per document.

    **Page numbers, not a count, and the difference is load-bearing.** `_sync` deletes a
    changed document's vectors but not its old page images, so a 5-page revision shortened
    to 2 leaves `_page_3..5.png` behind. Against a count, losing `_page_1.png` from that
    document still totals 4 >= 2 and reads as complete - while page 1's hit is dropped on
    every query. That is precisely the silent half-corpus this whole path exists to catch,
    so the check has to know *which* pages are present.
    """
    numbers: dict[str, set[int]] = {}
    if not PAGE_IMAGES_DIR.is_dir():
        return numbers
    for path in PAGE_IMAGES_DIR.iterdir():
        if not path.is_file():
            continue
        match = _PAGE_IMAGE_RE.fullmatch(path.name)
        if match:
            numbers.setdefault(match["stem"], set()).add(int(match["page"]))
    return numbers


def missing_page_numbers(stem: str, page_count: int,
                         rendered: dict[str, set[int]]) -> list[int]:
    """Which of `stem`'s expected pages have no image, given a `page_image_numbers` map.

    A document's pages are always exactly 1..n (`ingest_pdf` numbers them off the rendered
    page list), so the expected set is derivable and needs no extra bookkeeping. Extra
    images beyond `page_count` are ignored - a leftover PNG from a longer previous revision
    is cosmetic and breaks no query.

    Shared by `vector_store.index_health` (reporting) and `ingest._sync` (the repair
    decision) so the two can never disagree about what "complete" means.
    """
    present = rendered.get(stem, set())
    return [n for n in range(1, page_count + 1) if n not in present]


def crop_images_for(pdf_name: str) -> list[Path]:
    """Every answer-time crop/annotated PNG derived from one PDF's pages.

    Mirrors the names `highlight.crop_region` / `highlight.annotate_page` write:
    `<pdf-stem>_page_<n>_crop_<i>.png` and `<pdf-stem>_page_<n>_annotated.png`.
    Same anchoring rationale as `page_images_for`.
    """
    stem = re.escape(Path(pdf_name).stem)
    return _matching(CROPS_DIR, re.compile(rf"{stem}_page_\d+_(?:crop_\d+|annotated)\.png"))