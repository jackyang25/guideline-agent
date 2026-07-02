import json
import os

from .models import Manifest, PageRecord


def image_filename(pdf_index: int) -> str:
    return f"pages/p{pdf_index:03d}.png"


def record_filename(pdf_index: int) -> str:
    return f"pages/p{pdf_index:03d}.json"


def _write_text(base_dir: str, rel_path: str, text: str) -> None:
    full = os.path.join(base_dir, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(text)


def save_image(base_dir: str, pdf_index: int, image_bytes: bytes) -> str:
    rel = image_filename(pdf_index)
    full = os.path.join(base_dir, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(image_bytes)
    return rel


def write_page_record(base_dir: str, record: PageRecord) -> None:
    _write_text(
        base_dir,
        record_filename(record.pdf_index),
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
    )


def write_manifest(base_dir: str, manifest: Manifest) -> None:
    _write_text(
        base_dir,
        "manifest.json",
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
    )
