import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    on_page: Callable[[int, int], None] | None = None,
) -> tuple[Manifest, list[int]]:
    pages = rendered if rendered is not None else render_pdf(pdf_path)
    if limit is not None:
        pages = pages[:limit]

    page_numbers, flags = resolve_page_numbers(
        [p.raw_text for p in pages], [p.pdf_index for p in pages]
    )

    if client is None and describe_fn is describe_page:
        client = anthropic.Anthropic()

    total = len(pages)
    if on_page is not None:
        on_page(0, total)

    # Pages are independent, so describe them concurrently. Effective concurrency is
    # bounded by the account's token-per-minute limit; the SDK retries 429s with
    # backoff. Results come back as they finish (for live progress) but are
    # reassembled in page order for the manifest.
    entries: dict[int, PageMapEntry] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {
            executor.submit(
                _process_page, out_dir, guideline_id, page, page_number, describe_fn, client
            ): page.pdf_index
            for page, page_number in zip(pages, page_numbers)
        }
        done = 0
        for future in as_completed(futures):
            entries[futures[future]] = future.result()  # re-raises page failures
            done += 1
            if on_page is not None:
                on_page(done, total)

    map_entries = [entries[p.pdf_index] for p in pages]

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
