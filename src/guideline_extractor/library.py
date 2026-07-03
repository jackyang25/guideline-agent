import json
import os
from pathlib import Path


def root() -> Path:
    return Path(os.environ.get("GE_OUTPUT_ROOT", "guidelines")).resolve()


def _guideline_dir(guideline_id: str) -> Path:
    r = root()
    d = (r / guideline_id).resolve()
    if r not in d.parents and d != r:
        raise LookupError(f"invalid guideline id: {guideline_id!r}")
    if not (d / "manifest.json").is_file():
        raise LookupError(f"unknown guideline: {guideline_id!r}")
    return d


def list_guidelines() -> list[dict]:
    r = root()
    if not r.exists():
        return []
    out = []
    for child in r.iterdir():
        m = child / "manifest.json"
        if m.is_file():
            d = json.loads(m.read_text())
            out.append({
                "guideline_id": d["guideline_id"],
                "title": d["title"],
                "jurisdiction": d.get("jurisdiction"),
                "page_count": d["page_count"],
            })
    return sorted(out, key=lambda g: g["guideline_id"])


def load_manifest(guideline_id: str) -> dict:
    return json.loads((_guideline_dir(guideline_id) / "manifest.json").read_text())


def _pdf_index_for(manifest: dict, page_number: int) -> int | None:
    for p in manifest["pages"]:
        if p["page_number"] == page_number:
            return p["pdf_index"]
    return None


def load_page(guideline_id: str, page_number: int) -> dict | None:
    manifest = load_manifest(guideline_id)
    pdf_index = _pdf_index_for(manifest, page_number)
    if pdf_index is None:
        return None
    path = _guideline_dir(guideline_id) / "pages" / f"p{pdf_index:03d}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def neighbors(guideline_id: str, page_number: int) -> tuple[dict | None, dict | None]:
    pages = sorted(load_manifest(guideline_id)["pages"], key=lambda p: p["page_number"])
    numbers = [p["page_number"] for p in pages]
    if page_number not in numbers:
        return None, None
    i = numbers.index(page_number)
    def entry(j):
        p = pages[j]
        return {"page_number": p["page_number"], "title": p["title"]}
    return (entry(i - 1) if i > 0 else None, entry(i + 1) if i < len(pages) - 1 else None)
