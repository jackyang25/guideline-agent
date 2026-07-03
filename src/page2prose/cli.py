import argparse
import sys

from dotenv import load_dotenv

from .pipeline import extract

load_dotenv()  # read OPENAI_API_KEY from a .env file if present


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract prose-per-page from a guideline PDF.")
    parser.add_argument("pdf_path")
    parser.add_argument("out_root", help="Output root; the guideline is written to <out_root>/<id>.")
    parser.add_argument("--guideline-id", default=None,
                        help="Folder/citation key. Default: a slug of the title.")
    parser.add_argument("--guideline-title", default=None,
                        help="Default: detected from the cover page, else the filename.")
    parser.add_argument("--jurisdiction", default=None)
    parser.add_argument("--publisher", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--effective-date", default=None)
    parser.add_argument(
        "--concurrency", type=int, default=25,
        help="Pages described in parallel (default 25). Raise it as high as your "
             "token-per-minute limit sustains; the SDK retries 429s.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N pages (for a cheap smoke test).",
    )
    args = parser.parse_args(argv)

    manifest, flags, failed = extract(
        args.pdf_path,
        args.out_root,
        guideline_id=args.guideline_id,
        guideline_title=args.guideline_title,
        jurisdiction=args.jurisdiction,
        publisher=args.publisher,
        version=args.version,
        effective_date=args.effective_date,
        concurrency=args.concurrency,
        limit=args.limit,
    )
    print(f"Wrote {manifest.page_count} pages to {args.out_root}/{manifest.guideline_id}")
    print(f"  id: {manifest.guideline_id}  |  title: {manifest.title}")
    if flags:
        print(f"WARNING: page-number QC flagged page positions (0-based): {flags}")
    if failed:
        print(f"WARNING: {len(failed)} page(s) failed and were skipped (sheet indices {failed}). "
              f"Re-run to retry just those.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
