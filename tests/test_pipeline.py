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
