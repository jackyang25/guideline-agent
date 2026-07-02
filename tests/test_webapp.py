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


def test_extract_endpoint_wires_to_pipeline(out_root, monkeypatch):
    captured = {}

    def fake_extract(pdf_path, out_dir, guideline_id, **kw):
        captured["guideline_id"] = guideline_id
        captured["out_dir"] = out_dir
        captured["kw"] = kw
        m = Manifest(guideline_id, kw["guideline_title"], kw.get("jurisdiction"),
                     None, kw.get("version"), None, "x.pdf", 2,
                     [PageMapEntry(1, "t", 1), PageMapEntry(2, "u", 2)])
        return m, [1]

    monkeypatch.setattr(webapp, "extract", fake_extract)
    resp = client().post(
        "/api/extract",
        data={"guideline_id": "G1", "guideline_title": "Title", "limit": "2"},
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"guideline_id": "G1", "page_count": 2, "flags": [1]}
    assert captured["guideline_id"] == "G1"
    assert captured["kw"]["limit"] == 2


def test_index_page_served(out_root):
    resp = client().get("/")
    assert resp.status_code == 200
    assert "Guideline Extractor" in resp.text
