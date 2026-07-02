import argparse
import sys

from .pipeline import extract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract prose-per-page from a guideline PDF.")
    parser.add_argument("pdf_path")
    parser.add_argument("out_dir")
    parser.add_argument("--guideline-id", required=True)
    parser.add_argument("--guideline-title", required=True)
    parser.add_argument("--jurisdiction", default=None)
    parser.add_argument("--publisher", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--effective-date", default=None)
    args = parser.parse_args(argv)

    manifest, flags = extract(
        args.pdf_path,
        args.out_dir,
        args.guideline_id,
        guideline_title=args.guideline_title,
        jurisdiction=args.jurisdiction,
        publisher=args.publisher,
        version=args.version,
        effective_date=args.effective_date,
    )
    print(f"Wrote {manifest.page_count} pages to {args.out_dir}")
    if flags:
        print(f"WARNING: page-number QC flagged page positions (0-based): {flags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
