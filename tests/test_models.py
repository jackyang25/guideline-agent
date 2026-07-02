# tests/test_models.py
from guideline_extractor.models import PageRecord, PageMapEntry, Manifest


def test_page_record_to_dict_round_trips_fields():
    rec = PageRecord(
        guideline_id="APC_2023_ZA",
        page_number=38,
        pdf_index=40,
        title="TB treatment",
        prose="## TB treatment\n...",
        image_path="pages/p040.png",
        raw_text="verbatim",
    )
    assert rec.to_dict() == {
        "guideline_id": "APC_2023_ZA",
        "page_number": 38,
        "pdf_index": 40,
        "title": "TB treatment",
        "prose": "## TB treatment\n...",
        "image_path": "pages/p040.png",
        "raw_text": "verbatim",
    }


def test_manifest_to_dict_includes_page_map():
    m = Manifest(
        guideline_id="APC_2023_ZA",
        title="Adult Primary Care 2023",
        jurisdiction="South Africa",
        publisher=None,
        version="2023",
        effective_date=None,
        source_file="APC.pdf",
        page_count=1,
        pages=[PageMapEntry(page_number=1, title="Preface", pdf_index=1)],
    )
    d = m.to_dict()
    assert d["page_count"] == 1
    assert d["pages"] == [{"page_number": 1, "title": "Preface", "pdf_index": 1}]
    assert d["jurisdiction"] == "South Africa"
