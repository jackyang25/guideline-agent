import json

from page2prose.models import Manifest, PageMapEntry, PageRecord
from page2prose.storage import (
    image_filename,
    record_filename,
    save_image,
    write_page_record,
    write_manifest,
)


def test_filenames_zero_pad_to_three():
    assert image_filename(7) == "pages/p007.png"
    assert record_filename(40) == "pages/p040.json"


def test_save_image_writes_bytes_and_returns_relative_path(tmp_path):
    rel = save_image(str(tmp_path), 3, b"\x89PNG")
    assert rel == "pages/p003.png"
    assert (tmp_path / "pages" / "p003.png").read_bytes() == b"\x89PNG"


def test_write_page_record_serializes_by_pdf_index(tmp_path):
    rec = PageRecord("g", 38, 40, "T", "P", "pages/p040.png", "raw")
    write_page_record(str(tmp_path), rec)
    written = json.loads((tmp_path / "pages" / "p040.json").read_text())
    assert written["page_number"] == 38
    assert written["pdf_index"] == 40


def test_write_manifest_writes_json(tmp_path):
    m = Manifest("g", "T", "ZA", None, "2023", None, "a.pdf", 1,
                 [PageMapEntry(1, "Preface", 1)])
    write_manifest(str(tmp_path), m)
    written = json.loads((tmp_path / "manifest.json").read_text())
    assert written["guideline_id"] == "g"
    assert written["pages"][0]["title"] == "Preface"
