import anthropic

from .describe import describe_page
from .models import Manifest, PageMapEntry, PageRecord
from .pagemap import (
    assign_page_numbers,
    calibrate_offset,
    check_monotonic,
    detect_printed_number,
)
from .render import render_pdf
from .storage import save_image, write_manifest, write_page_record


def extract(
    pdf_path: str,
    out_dir: str,
    guideline_id: str,
    *,
    guideline_title: str,
    jurisdiction: str | None = None,
    publisher: str | None = None,
    version: str | None = None,
    effective_date: str | None = None,
    client=None,
    describe_fn=describe_page,
    rendered=None,
) -> tuple[Manifest, list[int]]:
    pages = rendered if rendered is not None else render_pdf(pdf_path)

    # Calibrate printed page numbers against sheet indices.
    samples = [(p.pdf_index, detect_printed_number(p.raw_text)) for p in pages]
    offset = calibrate_offset(samples)
    page_numbers = assign_page_numbers([p.pdf_index for p in pages], offset)

    # QC: flag sheets where the *printed* numbering itself breaks (not
    # strictly increasing), independent of the calibrated offset. Pages
    # with no detected printed number are skipped for this check, but
    # flag indices refer back to their position in `pages`.
    printed = [detect_printed_number(p.raw_text) for p in pages]
    present = [(i, n) for i, n in enumerate(printed) if n is not None]
    if present:
        sub_flags = check_monotonic([n for _, n in present])
        flags = [present[i][0] for i in sub_flags]
    else:
        flags = []

    if client is None and describe_fn is describe_page:
        client = anthropic.Anthropic()

    map_entries: list[PageMapEntry] = []
    for page, page_number in zip(pages, page_numbers):
        title, prose = describe_fn(client, page.image_bytes, page.raw_text)
        rel_image = save_image(out_dir, page.pdf_index, page.image_bytes)
        record = PageRecord(
            guideline_id=guideline_id,
            page_number=page_number,
            pdf_index=page.pdf_index,
            title=title,
            prose=prose,
            image_path=rel_image,
            raw_text=page.raw_text,
        )
        write_page_record(out_dir, record)
        map_entries.append(
            PageMapEntry(page_number=page_number, title=title, pdf_index=page.pdf_index)
        )

    manifest = Manifest(
        guideline_id=guideline_id,
        title=guideline_title,
        jurisdiction=jurisdiction,
        publisher=publisher,
        version=version,
        effective_date=effective_date,
        source_file=pdf_path.rsplit("/", 1)[-1],
        page_count=len(pages),
        pages=map_entries,
    )
    write_manifest(out_dir, manifest)
    return manifest, flags
