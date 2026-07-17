"""Geometry tests for the crop/annotate step (src.highlight).

These are pure-Pillow and need no models or API key, so they run fast.
"""

from PIL import Image

import src.highlight as highlight
from src.highlight import _to_pixel_box


def test_full_page_box_clamps_to_image():
    # [ymin, xmin, ymax, xmax] covering the whole 0-1000 space -> full image.
    assert _to_pixel_box([0, 0, 1000, 1000], 800, 1000) == (0, 0, 800, 1000)


def test_empty_or_malformed_box_falls_back_to_full_page():
    assert _to_pixel_box([], 800, 1000) == (0, 0, 800, 1000)
    assert _to_pixel_box([1, 2, 3], 800, 1000) == (0, 0, 800, 1000)


def test_mid_box_scales_and_pads():
    # [250,250,750,750] on a 1000x1000 page -> 250..750px, +4% (40px) padding.
    assert _to_pixel_box([250, 250, 750, 750], 1000, 1000) == (210, 210, 790, 790)


def test_swapped_min_max_is_normalized():
    # A box with max < min must still produce the same valid region.
    assert _to_pixel_box([750, 750, 250, 250], 1000, 1000) == (210, 210, 790, 790)


def test_padding_is_clamped_at_edges():
    # A box hugging the top-left corner cannot pad below (0, 0).
    left, upper, right, lower = _to_pixel_box([0, 0, 100, 100], 1000, 1000)
    assert (left, upper) == (0, 0)
    assert right <= 1000 and lower <= 1000


def test_degenerate_box_falls_back_to_full_page():
    # Zero-area box with no padding -> whole page rather than an empty crop.
    assert _to_pixel_box([500, 500, 500, 500], 1000, 1000, pad_frac=0) == (0, 0, 1000, 1000)


def test_crop_and_annotate_write_files(tmp_path, monkeypatch):
    monkeypatch.setattr(highlight, "CROPS_DIR", tmp_path)
    page = tmp_path / "doc_page_1.png"
    Image.new("RGB", (1000, 1400), "white").save(page)

    crop_path = highlight.crop_region(page, [200, 100, 400, 600])
    annotated_path = highlight.annotate_page(page, [200, 100, 400, 600])

    assert crop_path.exists() and annotated_path.exists()
    with Image.open(crop_path) as crop:
        # The crop is a strict sub-region of the page.
        assert crop.width < 1000 and crop.height < 1400
    with Image.open(annotated_path) as annotated:
        # The annotated copy keeps the full page dimensions.
        assert annotated.size == (1000, 1400)
