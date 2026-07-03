import json
import pytest
from page2prose.agent import tools


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path))
    tools._INDEX_CACHE.clear()
    d = tmp_path / "APC"
    (d / "pages").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "guideline_id": "APC", "title": "APC 2023", "jurisdiction": "South Africa",
        "publisher": None, "version": "2023", "effective_date": None,
        "source_file": "a.pdf", "page_count": 2,
        "pages": [
            {"page_number": 10, "title": "Cough", "pdf_index": 1},
            {"page_number": 11, "title": "TB treatment", "pdf_index": 2},
        ],
    }))
    (d / "pages" / "p001.json").write_text(json.dumps({
        "guideline_id": "APC", "page_number": 10, "pdf_index": 1, "title": "Cough",
        "prose": "Assess cough. Night sweats and weight loss suggest TB screening.",
        "image_path": "pages/p001.png", "raw_text": "r"}))
    (d / "pages" / "p002.json").write_text(json.dumps({
        "guideline_id": "APC", "page_number": 11, "pdf_index": 2, "title": "TB treatment",
        "prose": "Start the rifampicin isoniazid regimen once Xpert is positive.",
        "image_path": "pages/p002.png", "raw_text": "r"}))
    return tmp_path


def test_search_pages_ranks_body_content(lib):
    hits = tools.search_pages("APC", "night sweats weight loss")
    assert hits[0]["page_number"] == 10
    assert "title" in hits[0] and "snippet" in hits[0]


def test_search_pages_finds_treatment(lib):
    hits = tools.search_pages("APC", "rifampicin regimen")
    assert hits[0]["page_number"] == 11


def test_search_pages_no_match_is_empty(lib):
    assert tools.search_pages("APC", "cardiology echocardiogram angioplasty") == []


def test_get_page_returns_prose_and_neighbors(lib):
    p = tools.get_page("APC", 10)
    assert p["prose"].startswith("Assess cough")
    assert p["prev"] is None
    assert p["next"] == {"page_number": 11, "title": "TB treatment"}


def test_get_page_unknown_is_error(lib):
    assert "error" in tools.get_page("APC", 999)


def test_run_read_tool_dispatch(lib):
    assert tools.run_read_tool("list_guidelines", {})[0]["guideline_id"] == "APC"
    assert "error" in tools.run_read_tool("nope", {})


def test_tool_specs_cover_all_four():
    names = {t["function"]["name"] for t in tools.TOOL_SPECS}
    assert names == {"list_guidelines", "search_pages", "get_page", "submit_answer"}
