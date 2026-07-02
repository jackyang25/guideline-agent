import json

from guideline_extractor.render import RenderedPage
from guideline_extractor.pipeline import extract


def _fake_describe(client, image_bytes, raw_text):
    # title echoes the raw text's first line so assertions are meaningful
    return raw_text.splitlines()[0], f"prose for {raw_text.splitlines()[0]}"


def test_extract_writes_manifest_and_one_record_per_page(tmp_path):
    rendered = [
        RenderedPage(1, b"\x89PNG1", "Cough\n1"),
        RenderedPage(2, b"\x89PNG2", "TB treatment\n2"),
    ]
    manifest, flags = extract(
        "ignored.pdf",
        str(tmp_path),
        "APC_2023_ZA",
        guideline_title="Adult Primary Care 2023",
        jurisdiction="South Africa",
        describe_fn=_fake_describe,
        rendered=rendered,
    )
    assert manifest.page_count == 2
    assert flags == []
    # page numbers calibrated: printed 1,2 on sheets 1,2 -> offset 0
    assert [p.page_number for p in manifest.pages] == [1, 2]
    assert manifest.pages[1].title == "TB treatment"

    rec = json.loads((tmp_path / "pages" / "p002.json").read_text())
    assert rec["page_number"] == 2
    assert rec["title"] == "TB treatment"
    assert rec["image_path"] == "pages/p002.png"
    assert (tmp_path / "pages" / "p002.png").read_bytes() == b"\x89PNG2"
    assert (tmp_path / "manifest.json").exists()


def test_extract_flags_broken_numbering(tmp_path):
    # printed numbers 5 then 3 -> not strictly increasing -> flag index 1
    rendered = [
        RenderedPage(1, b"a", "A\n5"),
        RenderedPage(2, b"b", "B\n3"),
    ]
    _, flags = extract(
        "ignored.pdf", str(tmp_path), "g",
        guideline_title="t", describe_fn=_fake_describe, rendered=rendered,
    )
    assert flags == [1]


def test_extract_flags_printed_number_mismatch_with_calibrated_offset(tmp_path):
    # Sheets 1,2,3 with printed numbers 1,2,10. Calibration picks offset 0
    # (majority: two of three deltas are 0), so page 3 (printed 10) disagrees
    # with its calibrated page_number of 3 -> flagged, even though 1,2,10 is
    # still monotonically increasing.
    rendered = [
        RenderedPage(1, b"a", "A\n1"),
        RenderedPage(2, b"b", "B\n2"),
        RenderedPage(3, b"c", "C\n10"),
    ]
    _, flags = extract(
        "ignored.pdf", str(tmp_path), "g",
        guideline_title="t", describe_fn=_fake_describe, rendered=rendered,
    )
    assert flags == [2]


def test_extract_names_records_by_pdf_index_even_with_nonzero_offset(tmp_path):
    # printed 40, 41 on sheets 1, 2 -> offset 39 -> page_numbers 40, 41,
    # but files are still named by pdf_index (p001, p002).
    rendered = [
        RenderedPage(1, b"a", "A\n40"),
        RenderedPage(2, b"b", "B\n41"),
    ]
    manifest, flags = extract(
        "ignored.pdf", str(tmp_path), "g",
        guideline_title="t", describe_fn=_fake_describe, rendered=rendered,
    )
    assert flags == []
    assert [p.page_number for p in manifest.pages] == [40, 41]

    rec1 = json.loads((tmp_path / "pages" / "p001.json").read_text())
    assert rec1["page_number"] == 40
    rec2 = json.loads((tmp_path / "pages" / "p002.json").read_text())
    assert rec2["page_number"] == 41
