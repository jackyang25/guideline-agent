import os
from concurrent.futures import ThreadPoolExecutor

import anthropic

from .describe import describe_page
from .models import Manifest, PageMapEntry, PageRecord
from .pagemap import resolve_page_numbers
from .render import render_pdf
from .storage import save_image, write_manifest, write_page_record


def _process_page(out_dir, guideline_id, page, page_number, describe_fn, client) -> PageMapEntry:
    """Describe one page and write its record + image. Runs per worker thread."""
    title, prose = describe_fn(client, page.image_bytes, page.raw_text)
    rel_image = save_image(out_dir, page.pdf_index, page.image_bytes)
    write_page_record(
        out_dir,
        PageRecord(
            guideline_id=guideline_id,
            page_number=page_number,
            pdf_index=page.pdf_index,
            title=title,
            prose=prose,
            image_path=rel_image,
            raw_text=page.raw_text,
        ),
    )
    return PageMapEntry(page_number=page_number, title=title, pdf_index=page.pdf_index)


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
    concurrency: int = 25,
    limit: int | None = None,
) -> tuple[Manifest, list[int]]:
    pages = rendered if rendered is not None else render_pdf(pdf_path)
    if limit is not None:
        pages = pages[:limit]

    page_numbers, flags = resolve_page_numbers(
        [p.raw_text for p in pages], [p.pdf_index for p in pages]
    )

    if client is None and describe_fn is describe_page:
        client = anthropic.Anthropic()

    # Pages are independent, so describe them concurrently. ThreadPoolExecutor.map
    # preserves input order, so map_entries stays in page order regardless of which
    # worker finishes first. Effective concurrency is bounded by the account's
    # token-per-minute limit; the SDK retries 429s with backoff.
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        map_entries: list[PageMapEntry] = list(
            executor.map(
                lambda pair: _process_page(
                    out_dir, guideline_id, pair[0], pair[1], describe_fn, client
                ),
                list(zip(pages, page_numbers)),
            )
        )

    manifest = Manifest(
        guideline_id=guideline_id,
        title=guideline_title,
        jurisdiction=jurisdiction,
        publisher=publisher,
        version=version,
        effective_date=effective_date,
        source_file=os.path.basename(pdf_path),
        page_count=len(pages),
        pages=map_entries,
    )
    write_manifest(out_dir, manifest)
    return manifest, flags
