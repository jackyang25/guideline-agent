import fitz  # PyMuPDF
import pytest
from guideline_extractor.render import render_pdf, RenderedPage


@pytest.fixture
def sample_pdf(tmp_path):
    doc = fitz.open()
    for text in ["Page one body\n1", "Page two body\n2"]:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    path = tmp_path / "sample.pdf"
    doc.save(path)
    doc.close()
    return str(path)


def test_render_pdf_returns_one_rendered_page_per_sheet(sample_pdf):
    pages = render_pdf(sample_pdf)
    assert len(pages) == 2
    assert all(isinstance(p, RenderedPage) for p in pages)


def test_render_pdf_indices_are_1_based_and_ordered(sample_pdf):
    pages = render_pdf(sample_pdf)
    assert [p.pdf_index for p in pages] == [1, 2]


def test_render_pdf_extracts_text_and_png_bytes(sample_pdf):
    pages = render_pdf(sample_pdf)
    assert "Page one body" in pages[0].raw_text
    assert pages[0].image_bytes.startswith(b"\x89PNG")
