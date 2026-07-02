from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class RenderedPage:
    pdf_index: int  # 1-based sheet position
    image_bytes: bytes
    raw_text: str


def render_pdf(pdf_path: str, dpi: int = 150) -> list[RenderedPage]:
    doc = fitz.open(pdf_path)
    try:
        pages: list[RenderedPage] = []
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi)
            pages.append(
                RenderedPage(
                    pdf_index=i,
                    image_bytes=pix.tobytes("png"),
                    raw_text=page.get_text("text"),
                )
            )
        return pages
    finally:
        doc.close()
