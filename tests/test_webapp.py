import json

import pytest
from fastapi.testclient import TestClient

from guideline_extractor import webapp
from guideline_extractor.models import Manifest, PageMapEntry


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def seeded(out_root):
    d = out_root / "APC_2023_ZA"
    (d / "pages").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "guideline_id": "APC_2023_ZA",
        "title": "Adult Primary Care 2023",
        "jurisdiction": "South Africa",
        "version": "2023",
        "page_count": 1,
        "pages": [{"page_number": 34, "title": "Cough", "pdf_index": 1}],
    }))
    (d / "pages" / "p001.json").write_text(json.dumps({
        "guideline_id": "APC_2023_ZA",
        "page_number": 34,
        "pdf_index": 1,
        "title": "Cough",
        "prose": "## Cough\n\n| Drug | Dose |\n|---|---|\n| Amoxicillin | 500 mg |",
        "image_path": "pages/p001.png",
        "raw_text": "Cough\n34",
    }))
    (d / "pages" / "p001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return out_root


def client():
    return TestClient(webapp.app)


def test_list_guidelines_empty(out_root):
    assert client().get("/api/guidelines").json() == []


def test_list_guidelines_returns_seeded(seeded):
    data = client().get("/api/guidelines").json()
    assert data == [{"guideline_id": "APC_2023_ZA", "title": "Adult Primary Care 2023", "page_count": 1}]


def test_get_manifest(seeded):
    m = client().get("/api/guidelines/APC_2023_ZA/manifest").json()
    assert m["pages"][0]["page_number"] == 34


def test_get_page_renders_prose_markdown_to_html(seeded):
    r = client().get("/api/guidelines/APC_2023_ZA/pages/1").json()
    assert r["page_number"] == 34
    assert r["raw_text"] == "Cough\n34"
    # prose markdown table rendered to an HTML table, raw prose preserved
    assert "<table>" in r["prose_html"]
    assert r["prose"].startswith("## Cough")


def test_get_image_returns_png(seeded):
    resp = client().get("/api/guidelines/APC_2023_ZA/image/1")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG")


def test_missing_guideline_is_404(out_root):
    assert client().get("/api/guidelines/nope/manifest").status_code == 404


def test_path_traversal_rejected(seeded):
    assert client().get("/api/guidelines/..%2f..%2fetc/manifest").status_code in (400, 404)


def test_extract_endpoint_streams_progress_then_done(out_root, monkeypatch):
    captured = {}

    def fake_extract(pdf_path, out_dir, guideline_id, on_page=None, **kw):
        captured["guideline_id"] = guideline_id
        captured["kw"] = kw
        total = 2
        if on_page:
            on_page(0, total)
            on_page(1, total)
            on_page(2, total)
        m = Manifest(guideline_id, kw["guideline_title"], kw.get("jurisdiction"),
                     None, kw.get("version"), None, "x.pdf", total,
                     [PageMapEntry(1, "t", 1), PageMapEntry(2, "u", 2)])
        return m, [1]

    monkeypatch.setattr(webapp, "extract", fake_extract)
    resp = client().post(
        "/api/extract",
        data={"guideline_id": "G1", "guideline_title": "Title", "limit": "2"},
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    progress = [e for e in events if e["type"] == "progress"]
    assert progress[-1] == {"type": "progress", "done": 2, "total": 2}
    assert events[-1] == {"type": "done", "guideline_id": "G1", "page_count": 2, "flags": [1]}
    assert captured["kw"]["limit"] == 2


def test_extract_endpoint_streams_error(out_root, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no api key")

    monkeypatch.setattr(webapp, "extract", boom)
    resp = client().post(
        "/api/extract",
        data={"guideline_id": "G1", "guideline_title": "Title"},
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert events[-1]["type"] == "error"
    assert "no api key" in events[-1]["detail"]


def test_index_page_served(out_root):
    resp = client().get("/")
    assert resp.status_code == 200
    assert "Guideline Extractor" in resp.text


def test_index_js_has_no_unescaped_newline_in_string_literal():
    # The embedded JS must contain the two-char sequence backslash-n, not a real
    # newline (which would be a JS syntax error and break every handler).
    assert r"buf.indexOf('\n')" in webapp.INDEX_HTML
    assert "buf.indexOf('\n')" not in webapp.INDEX_HTML  # literal newline = bug
