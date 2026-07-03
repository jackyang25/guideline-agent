import json
import pytest
from guideline_extractor import library


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path))
    d = tmp_path / "APC"
    (d / "pages").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "guideline_id": "APC", "title": "APC 2023", "jurisdiction": "South Africa",
        "publisher": None, "version": "2023", "effective_date": None,
        "source_file": "a.pdf", "page_count": 2,
        "pages": [
            {"page_number": 10, "title": "Cough", "pdf_index": 1},
            {"page_number": 11, "title": "TB", "pdf_index": 2},
        ],
    }))
    (d / "pages" / "p001.json").write_text(json.dumps({
        "guideline_id": "APC", "page_number": 10, "pdf_index": 1, "title": "Cough",
        "prose": "cough prose", "image_path": "pages/p001.png", "raw_text": "raw1"}))
    (d / "pages" / "p002.json").write_text(json.dumps({
        "guideline_id": "APC", "page_number": 11, "pdf_index": 2, "title": "TB",
        "prose": "tb prose", "image_path": "pages/p002.png", "raw_text": "raw2"}))
    return tmp_path


def test_list_guidelines(lib):
    assert library.list_guidelines() == [
        {"guideline_id": "APC", "title": "APC 2023", "jurisdiction": "South Africa", "page_count": 2}
    ]


def test_list_guidelines_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path / "nope"))
    assert library.list_guidelines() == []


def test_load_page_by_printed_number(lib):
    rec = library.load_page("APC", 11)
    assert rec["pdf_index"] == 2
    assert rec["prose"] == "tb prose"


def test_load_page_missing_returns_none(lib):
    assert library.load_page("APC", 999) is None


def test_load_manifest_unknown_raises(lib):
    with pytest.raises(LookupError):
        library.load_manifest("../etc")


def test_neighbors(lib):
    prev, nxt = library.neighbors("APC", 10)
    assert prev is None
    assert nxt == {"page_number": 11, "title": "TB"}
    prev, nxt = library.neighbors("APC", 11)
    assert prev == {"page_number": 10, "title": "Cough"}
    assert nxt is None


def test_list_guidelines_sorted_by_guideline_id(tmp_path, monkeypatch):
    """Regression: list_guidelines() must sort by guideline_id, not directory name."""
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path))

    # Create two guidelines where directory name sorts opposite to guideline_id
    # Directory 'z_dir' has guideline_id 'AAA' (should come first when sorted by ID)
    z_dir = tmp_path / "z_dir"
    z_dir.mkdir()
    (z_dir / "manifest.json").write_text(json.dumps({
        "guideline_id": "AAA",
        "title": "First Guideline",
        "jurisdiction": "Region A",
        "page_count": 1,
    }))

    # Directory 'a_dir' has guideline_id 'ZZZ' (should come second when sorted by ID)
    a_dir = tmp_path / "a_dir"
    a_dir.mkdir()
    (a_dir / "manifest.json").write_text(json.dumps({
        "guideline_id": "ZZZ",
        "title": "Last Guideline",
        "jurisdiction": "Region Z",
        "page_count": 1,
    }))

    # list_guidelines() should return sorted by guideline_id, not directory name
    result = library.list_guidelines()
    assert len(result) == 2
    assert result[0]["guideline_id"] == "AAA"
    assert result[1]["guideline_id"] == "ZZZ"
