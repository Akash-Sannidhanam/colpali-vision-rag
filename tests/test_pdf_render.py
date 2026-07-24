"""Tests for the page-image naming helpers (src.pdf_render).

Pure filename logic - no poppler, no PIL, no rendering. These back document deletion,
where matching one filename too many is unrecoverable, so the anchoring is asserted
directly rather than assumed.
"""

from src import pdf_render


def _touch(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"x")


def _at(monkeypatch, pages, crops):
    monkeypatch.setattr(pdf_render, "PAGE_IMAGES_DIR", pages)
    monkeypatch.setattr(pdf_render, "CROPS_DIR", crops)


def test_page_image_path_is_the_naming_source_of_truth(tmp_path, monkeypatch):
    _at(monkeypatch, tmp_path, tmp_path / "crops")
    assert pdf_render.page_image_path("sales_report.pdf", 3).name == "sales_report_page_3.png"


def test_page_images_for_collects_every_page_of_one_document(tmp_path, monkeypatch):
    pages = tmp_path / "page_images"
    _touch(pages, "a_page_1.png", "a_page_2.png", "a_page_10.png", "b_page_1.png")
    _at(monkeypatch, pages, tmp_path / "crops")

    assert [p.name for p in pdf_render.page_images_for("a.pdf")] == [
        "a_page_1.png", "a_page_10.png", "a_page_2.png",
    ]


def test_page_images_for_does_not_bleed_into_a_similarly_named_document(tmp_path, monkeypatch):
    # `report_page_1.pdf` renders to `report_page_1_page_1.png`. A `report_page_*` glob
    # would sweep that up while deleting `report.pdf` - the anchored match must not.
    pages = tmp_path / "page_images"
    _touch(pages, "report_page_1.png", "report_page_1_page_1.png", "report_v2_page_1.png")
    _at(monkeypatch, pages, tmp_path / "crops")

    assert [p.name for p in pdf_render.page_images_for("report.pdf")] == ["report_page_1.png"]
    assert [p.name for p in pdf_render.page_images_for("report_page_1.pdf")] == [
        "report_page_1_page_1.png",
    ]


def test_crop_images_for_matches_crops_and_annotated_only(tmp_path, monkeypatch):
    crops = tmp_path / "crops"
    _touch(crops, "a_page_1_crop_0.png", "a_page_1_crop_11.png", "a_page_2_annotated.png",
           "a_page_1.png", "b_page_1_crop_0.png")
    _at(monkeypatch, tmp_path / "page_images", crops)

    assert [p.name for p in pdf_render.crop_images_for("a.pdf")] == [
        "a_page_1_crop_0.png", "a_page_1_crop_11.png", "a_page_2_annotated.png",
    ]


def test_helpers_return_empty_when_the_directory_is_absent(tmp_path, monkeypatch):
    _at(monkeypatch, tmp_path / "nope", tmp_path / "also-nope")
    assert pdf_render.page_images_for("a.pdf") == []
    assert pdf_render.crop_images_for("a.pdf") == []


def test_document_names_with_regex_metacharacters_are_escaped(tmp_path, monkeypatch):
    pages = tmp_path / "page_images"
    _touch(pages, "a.b_page_1.png", "axb_page_1.png")
    _at(monkeypatch, pages, tmp_path / "crops")

    # an unescaped "." would also match "axb_page_1.png"
    assert [p.name for p in pdf_render.page_images_for("a.b.pdf")] == ["a.b_page_1.png"]
