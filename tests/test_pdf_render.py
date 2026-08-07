"""Tests for the page-image naming helpers (src.pdf_render).

Pure filename logic - no poppler, no PIL, no rendering. These back document deletion,
where matching one filename too many is unrecoverable, so the anchoring is asserted
directly rather than assumed.
"""

from src import pdf_render


def _touch(directory, *names):
    """Create `directory` and write a stub byte into each named file."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"x")


def _at(monkeypatch, pages, crops):
    """Point the page-image and crop directories at tmp paths for one test."""
    monkeypatch.setattr(pdf_render, "PAGE_IMAGES_DIR", pages)
    monkeypatch.setattr(pdf_render, "CROPS_DIR", crops)


def test_page_image_path_is_the_naming_source_of_truth(tmp_path, monkeypatch):
    """The `<stem>_page_<n>.png` naming both the writer and readers depend on."""
    _at(monkeypatch, tmp_path, tmp_path / "crops")
    assert pdf_render.page_image_path("sales_report.pdf", 3).name == "sales_report_page_3.png"


def test_page_images_for_collects_every_page_of_one_document(tmp_path, monkeypatch):
    """Every page of the named document is found, and no other document's is."""
    pages = tmp_path / "page_images"
    _touch(pages, "a_page_1.png", "a_page_2.png", "a_page_10.png", "b_page_1.png")
    _at(monkeypatch, pages, tmp_path / "crops")

    assert [p.name for p in pdf_render.page_images_for("a.pdf")] == [
        "a_page_1.png", "a_page_10.png", "a_page_2.png",
    ]


def test_page_images_for_does_not_bleed_into_a_similarly_named_document(tmp_path, monkeypatch):
    """Anchoring keeps `report.pdf` from sweeping up `report_page_1.pdf`'s images."""
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
    """Crops and annotated pages match; a plain page image in the same dir does not."""
    crops = tmp_path / "crops"
    _touch(crops, "a_page_1_crop_0.png", "a_page_1_crop_11.png", "a_page_2_annotated.png",
           "a_page_1.png", "b_page_1_crop_0.png")
    _at(monkeypatch, tmp_path / "page_images", crops)

    assert [p.name for p in pdf_render.crop_images_for("a.pdf")] == [
        "a_page_1_crop_0.png", "a_page_1_crop_11.png", "a_page_2_annotated.png",
    ]


def test_helpers_return_empty_when_the_directory_is_absent(tmp_path, monkeypatch):
    """A missing directory yields [] rather than raising, so a cold install can delete."""
    _at(monkeypatch, tmp_path / "nope", tmp_path / "also-nope")
    assert pdf_render.page_images_for("a.pdf") == []
    assert pdf_render.crop_images_for("a.pdf") == []


def test_document_names_with_regex_metacharacters_are_escaped(tmp_path, monkeypatch):
    """A '.' in a document name is matched literally, not as a regex wildcard."""
    pages = tmp_path / "page_images"
    _touch(pages, "a.b_page_1.png", "axb_page_1.png")
    _at(monkeypatch, pages, tmp_path / "crops")

    # an unescaped "." would also match "axb_page_1.png"
    assert [p.name for p in pdf_render.page_images_for("a.b.pdf")] == ["a.b_page_1.png"]


# --- page_image_numbers: the bulk form of page_images_for ---

def test_page_image_numbers_records_which_pages_each_document_has(monkeypatch, tmp_path):
    """One directory scan instead of one per document (vector_store.index_health, ingest._sync)."""
    monkeypatch.setattr(pdf_render, "PAGE_IMAGES_DIR", tmp_path)
    for name in ("a_page_1.png", "a_page_2.png", "b_page_1.png"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")           # not a page image
    (tmp_path / "crops").mkdir()                          # not a file

    assert pdf_render.page_image_numbers() == {"a": {1, 2}, "b": {1}}


def test_page_image_numbers_agrees_with_page_images_for(monkeypatch, tmp_path):
    """The ambiguous filename that anchoring exists for, checked against the per-stem form.

    `report_page_1.pdf` renders to `report_page_1_page_2.png`, which a loose `report_*`
    match would credit to `report.pdf`. The bulk parser's greedy stem must make the same
    call as `page_images_for`'s anchored regex, or index_health would report a phantom
    complete document and _sync would skip a broken one.
    """
    monkeypatch.setattr(pdf_render, "PAGE_IMAGES_DIR", tmp_path)
    for name in ("report_page_1.png", "report_page_1_page_1.png", "report_page_1_page_2.png"):
        (tmp_path / name).write_bytes(b"x")

    numbers = pdf_render.page_image_numbers()

    assert numbers == {"report": {1}, "report_page_1": {1, 2}}
    assert len(numbers["report"]) == len(pdf_render.page_images_for("report.pdf"))
    assert len(numbers["report_page_1"]) == len(pdf_render.page_images_for("report_page_1.pdf"))


def test_page_image_numbers_is_empty_when_the_directory_is_missing(monkeypatch, tmp_path):
    """A fresh checkout has no page_images/ yet; boot must not fail on that."""
    monkeypatch.setattr(pdf_render, "PAGE_IMAGES_DIR", tmp_path / "nope")

    assert pdf_render.page_image_numbers() == {}


# --- missing_page_numbers: the shared definition of "complete" ---

def test_missing_page_numbers_names_the_gaps():
    """A document's pages are always 1..n, so the expected set needs no bookkeeping."""
    assert pdf_render.missing_page_numbers("a", 4, {"a": {1, 3}}) == [2, 4]
    assert pdf_render.missing_page_numbers("a", 3, {"a": {1, 2, 3}}) == []
    assert pdf_render.missing_page_numbers("gone", 2, {}) == [1, 2]


def test_missing_page_numbers_ignores_leftovers_but_not_gaps():
    """The counting bug, at its source.

    `_sync` never deletes a changed document's old page images, so a shortened revision
    leaves high-numbered PNGs behind. A bare count of 4 against 2 indexed pages reads as
    complete while page 1 is actually gone - so the check is over page *numbers*, and
    extras past the count stay ignored because they break no query.
    """
    assert pdf_render.missing_page_numbers("a", 2, {"a": {2, 3, 4, 5}}) == [1]
    assert pdf_render.missing_page_numbers("a", 2, {"a": {1, 2, 3, 4, 5}}) == []
