"""Crop and annotate the page region a Gemini answer was read from.

Gemini returns boxes as [ymin, xmin, ymax, xmax] normalized to a 0-1000 scale
(top-left origin). We convert that to pixels against the real page image, pad a
little for context, and produce both a tight crop and a full page with the box
drawn on it.
"""

from pathlib import Path
from PIL import Image, ImageDraw

from src.config import CROPS_DIR


def _to_pixel_box(
    box: list[int], width: int, height: int, pad_frac: float = 0.04
) -> tuple[int, int, int, int]:
    """Convert a 0-1000 [ymin, xmin, ymax, xmax] box to a padded, clamped pixel
    (left, upper, right, lower) tuple. Falls back to the full page if the box is
    missing or degenerate."""
    if not box or len(box) != 4:
        return (0, 0, width, height)

    ymin, xmin, ymax, xmax = box
    left = xmin / 1000 * width
    right = xmax / 1000 * width
    upper = ymin / 1000 * height
    lower = ymax / 1000 * height

    # Normalize in case the model swapped min/max.
    left, right = min(left, right), max(left, right)
    upper, lower = min(upper, lower), max(upper, lower)

    pad_x = pad_frac * width
    pad_y = pad_frac * height
    left = int(max(0, left - pad_x))
    upper = int(max(0, upper - pad_y))
    right = int(min(width, right + pad_x))
    lower = int(min(height, lower + pad_y))

    # Degenerate box -> show the whole page rather than an empty crop.
    if right <= left or lower <= upper:
        return (0, 0, width, height)
    return (left, upper, right, lower)


def crop_region(image_path: Path, box: list[int]) -> Path:
    """Crop the answer region out of a page PNG and save it. Returns the path."""
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    with Image.open(image_path) as page:
        page = page.convert("RGB")
        pixel_box = _to_pixel_box(box, page.width, page.height)
        crop = page.crop(pixel_box)
        out_path = CROPS_DIR / f"{image_path.stem}_crop.png"
        crop.save(out_path, format="PNG")
    return out_path


def annotate_page(image_path: Path, box: list[int]) -> Path:
    """Save a copy of the page with the answer region outlined. Returns the path."""
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    with Image.open(image_path) as page:
        page = page.convert("RGB")
        pixel_box = _to_pixel_box(box, page.width, page.height)
        draw = ImageDraw.Draw(page)
        draw.rectangle(pixel_box, outline="red", width=max(3, page.width // 200))
        out_path = CROPS_DIR / f"{image_path.stem}_annotated.png"
        page.save(out_path, format="PNG")
    return out_path
