# Guideline Extraction Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python pipeline that turns a guideline PDF into one prose-per-page record set (`manifest.json` + `pages/pNNN.{json,png}`) that stands in for the pages and supports downstream agent navigation.

**Architecture:** Six focused modules. `render` turns the PDF into per-page images + raw text (PyMuPDF). `pagemap` calibrates printed page numbers against sheet indices with a QC gate. `describe` sends image + raw text to Claude Opus 4.8 with the fidelity-contract prompt and gets back `{title, prose}`. `storage` writes the on-disk layout. `pipeline` orchestrates; `cli` is the entrypoint. Deterministic logic (pagemap, models, storage) is unit-tested directly; IO boundaries (render, describe) are tested with a generated fixture PDF and a fake Claude client.

**Tech Stack:** Python 3.11+, PyMuPDF (`pymupdf`), `anthropic` SDK, `pytest`.

## Global Constraints

- Model ID is exactly `claude-opus-4-8` (Claude Opus 4.8) — never a date-suffixed variant.
- Vision calls use `client.messages.stream(...)` + `get_final_message()` (prose can be long; streaming avoids SDK HTTP timeouts).
- Adaptive thinking only: `thinking={"type": "adaptive"}`. Never pass `budget_tokens`, `temperature`, `top_p`, or `top_k` (all 400 on Opus 4.8).
- Structured output via `output_config={"format": {"type": "json_schema", "schema": ...}}` — never the deprecated `output_format`.
- The physical page is the addressable unit — one record + one PNG per PDF sheet. No merging/splitting.
- `page_number` = printed number (calibrated); `pdf_index` = 1-based sheet position. On-disk filenames use zero-padded `pdf_index` (`p001`) — guaranteed unique and calibration-independent; the manifest map resolves `page_number` → sheet.
- All secrets from env: construct `anthropic.Anthropic()` with no arguments.

---

### Task 1: Project scaffold + data models

**Files:**
- Create: `pyproject.toml`
- Create: `src/guideline_extractor/__init__.py`
- Create: `src/guideline_extractor/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `PageRecord(guideline_id: str, page_number: int, pdf_index: int, title: str, prose: str, image_path: str, raw_text: str)` with `to_dict() -> dict`.
  - `PageMapEntry(page_number: int, title: str, pdf_index: int)` with `to_dict() -> dict`.
  - `Manifest(guideline_id: str, title: str, jurisdiction: str | None, publisher: str | None, version: str | None, effective_date: str | None, source_file: str, page_count: int, pages: list[PageMapEntry])` with `to_dict() -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from guideline_extractor.models import PageRecord, PageMapEntry, Manifest


def test_page_record_to_dict_round_trips_fields():
    rec = PageRecord(
        guideline_id="APC_2023_ZA",
        page_number=38,
        pdf_index=40,
        title="TB treatment",
        prose="## TB treatment\n...",
        image_path="pages/p040.png",
        raw_text="verbatim",
    )
    assert rec.to_dict() == {
        "guideline_id": "APC_2023_ZA",
        "page_number": 38,
        "pdf_index": 40,
        "title": "TB treatment",
        "prose": "## TB treatment\n...",
        "image_path": "pages/p040.png",
        "raw_text": "verbatim",
    }


def test_manifest_to_dict_includes_page_map():
    m = Manifest(
        guideline_id="APC_2023_ZA",
        title="Adult Primary Care 2023",
        jurisdiction="South Africa",
        publisher=None,
        version="2023",
        effective_date=None,
        source_file="APC.pdf",
        page_count=1,
        pages=[PageMapEntry(page_number=1, title="Preface", pdf_index=1)],
    )
    d = m.to_dict()
    assert d["page_count"] == 1
    assert d["pages"] == [{"page_number": 1, "title": "Preface", "pdf_index": 1}]
    assert d["jurisdiction"] == "South Africa"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guideline_extractor'`

- [ ] **Step 3: Write the scaffold and implementation**

```toml
# pyproject.toml
[project]
name = "guideline-extractor"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pymupdf>=1.24", "anthropic>=0.40"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```python
# src/guideline_extractor/__init__.py
```

```python
# src/guideline_extractor/models.py
from dataclasses import dataclass, asdict


@dataclass
class PageRecord:
    guideline_id: str
    page_number: int
    pdf_index: int
    title: str
    prose: str
    image_path: str
    raw_text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageMapEntry:
    page_number: int
    title: str
    pdf_index: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Manifest:
    guideline_id: str
    title: str
    jurisdiction: str | None
    publisher: str | None
    version: str | None
    effective_date: str | None
    source_file: str
    page_count: int
    pages: list[PageMapEntry]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pages"] = [p.to_dict() for p in self.pages]
        return d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[dev]" && pytest tests/test_models.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/guideline_extractor/__init__.py src/guideline_extractor/models.py tests/test_models.py
git commit -m "feat: project scaffold and data models"
```

---

### Task 2: Page-number calibration + QC gate

**Files:**
- Create: `src/guideline_extractor/pagemap.py`
- Test: `tests/test_pagemap.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `detect_printed_number(raw_text: str) -> int | None` — heuristic: the last standalone 1–4 digit line in the text (page footer), else None.
  - `calibrate_offset(samples: list[tuple[int, int | None]]) -> int | None` — samples are `(pdf_index, printed_or_None)`; returns the most common `printed - pdf_index`, or None if no printed numbers.
  - `assign_page_numbers(pdf_indices: list[int], offset: int | None) -> list[int]` — `page_number = pdf_index + offset`; if offset is None, `page_number = pdf_index`.
  - `check_monotonic(page_numbers: list[int]) -> list[int]` — 0-based indices where the sequence is not strictly increasing (QC flags).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pagemap.py
from guideline_extractor.pagemap import (
    detect_printed_number,
    calibrate_offset,
    assign_page_numbers,
    check_monotonic,
)


def test_detect_printed_number_returns_last_standalone_integer():
    text = "TB treatment\nStart HRZE regimen\n38"
    assert detect_printed_number(text) == 38


def test_detect_printed_number_ignores_numbers_inside_text():
    text = "Give 500 mg amoxicillin\nsome prose here"
    assert detect_printed_number(text) is None


def test_detect_printed_number_none_when_empty():
    assert detect_printed_number("") is None


def test_calibrate_offset_picks_most_common_delta():
    # printed = pdf_index - 2 on the pages that have a number
    samples = [(3, 1), (4, 2), (5, None), (6, 4)]
    assert calibrate_offset(samples) == -2


def test_calibrate_offset_none_when_no_numbers():
    assert calibrate_offset([(1, None), (2, None)]) is None


def test_assign_page_numbers_applies_offset():
    assert assign_page_numbers([3, 4, 5], -2) == [1, 2, 3]


def test_assign_page_numbers_falls_back_to_pdf_index():
    assert assign_page_numbers([1, 2, 3], None) == [1, 2, 3]


def test_check_monotonic_flags_breaks():
    # index 2 breaks (5 -> 4), index 3 is fine again (4 -> 9)
    assert check_monotonic([1, 5, 4, 9]) == [2]


def test_check_monotonic_clean_sequence_has_no_flags():
    assert check_monotonic([1, 2, 3]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pagemap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guideline_extractor.pagemap'`

- [ ] **Step 3: Write the implementation**

```python
# src/guideline_extractor/pagemap.py
import re
from collections import Counter

_STANDALONE_INT = re.compile(r"^\s*(\d{1,4})\s*$")


def detect_printed_number(raw_text: str) -> int | None:
    found: int | None = None
    for line in raw_text.splitlines():
        m = _STANDALONE_INT.match(line)
        if m:
            found = int(m.group(1))  # keep the last standalone int (footer)
    return found


def calibrate_offset(samples: list[tuple[int, int | None]]) -> int | None:
    deltas = [printed - idx for idx, printed in samples if printed is not None]
    if not deltas:
        return None
    return Counter(deltas).most_common(1)[0][0]


def assign_page_numbers(pdf_indices: list[int], offset: int | None) -> list[int]:
    if offset is None:
        return list(pdf_indices)
    return [idx + offset for idx in pdf_indices]


def check_monotonic(page_numbers: list[int]) -> list[int]:
    flags: list[int] = []
    for i in range(1, len(page_numbers)):
        if page_numbers[i] <= page_numbers[i - 1]:
            flags.append(i)
    return flags
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pagemap.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/guideline_extractor/pagemap.py tests/test_pagemap.py
git commit -m "feat: printed page-number calibration and QC gate"
```

---

### Task 3: PDF rendering (image + raw text)

**Files:**
- Create: `src/guideline_extractor/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `RenderedPage(pdf_index: int, image_bytes: bytes, raw_text: str)` (dataclass; `pdf_index` is 1-based).
  - `render_pdf(pdf_path: str, dpi: int = 150) -> list[RenderedPage]` — renders each page to PNG bytes and extracts the text layer.

- [ ] **Step 1: Write the failing test**

The fixture builds a tiny 2-page PDF with PyMuPDF so the test is self-contained.

```python
# tests/test_render.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guideline_extractor.render'`

- [ ] **Step 3: Write the implementation**

```python
# src/guideline_extractor/render.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_render.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/guideline_extractor/render.py tests/test_render.py
git commit -m "feat: PDF rendering to per-page image and raw text"
```

---

### Task 4: Fidelity-contract description via Claude

**Files:**
- Create: `src/guideline_extractor/describe.py`
- Test: `tests/test_describe.py`

**Interfaces:**
- Consumes: nothing from prior tasks (takes raw bytes + text).
- Produces:
  - `FIDELITY_PROMPT: str` — the generalized fidelity-contract instruction, adapted to request a JSON object.
  - `build_messages(image_bytes: bytes, raw_text: str) -> list[dict]` — the `messages` payload (image block then text block).
  - `parse_description(text: str) -> tuple[str, str]` — parses the JSON response into `(title, prose)`; raises `ValueError` on missing keys.
  - `describe_page(client, image_bytes: bytes, raw_text: str, model: str = "claude-opus-4-8", max_tokens: int = 16000) -> tuple[str, str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_describe.py
import base64
import json

import pytest
from guideline_extractor.describe import (
    build_messages,
    parse_description,
    describe_page,
    FIDELITY_PROMPT,
)


def test_build_messages_puts_base64_image_before_text():
    msgs = build_messages(b"\x89PNGdata", "raw ground truth")
    content = msgs[0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["data"] == base64.standard_b64encode(b"\x89PNGdata").decode()
    assert content[1]["type"] == "text"
    assert "raw ground truth" in content[1]["text"]
    assert FIDELITY_PROMPT in content[1]["text"]


def test_parse_description_extracts_title_and_prose():
    text = json.dumps({"title": "Cough", "prose": "## Cough\n..."})
    assert parse_description(text) == ("Cough", "## Cough\n...")


def test_parse_description_raises_on_missing_keys():
    with pytest.raises(ValueError):
        parse_description(json.dumps({"title": "x"}))


class _FakeStream:
    def __init__(self, text):
        self._text = text

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        block = type("Block", (), {"type": "text", "text": self._text})()
        return type("Msg", (), {"content": [block]})()


class _FakeMessages:
    def __init__(self, text):
        self._text = text
        self.captured = None

    def stream(self, **kwargs):
        self.captured = kwargs
        return _FakeStream(self._text)


class _FakeClient:
    def __init__(self, text):
        self.messages = _FakeMessages(text)


def test_describe_page_returns_title_and_prose_and_uses_opus():
    client = _FakeClient(json.dumps({"title": "T", "prose": "P"}))
    title, prose = describe_page(client, b"\x89PNG", "raw")
    assert (title, prose) == ("T", "P")
    assert client.messages.captured["model"] == "claude-opus-4-8"
    assert client.messages.captured["thinking"] == {"type": "adaptive"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_describe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guideline_extractor.describe'`

- [ ] **Step 3: Write the implementation**

```python
# src/guideline_extractor/describe.py
import base64
import json

FIDELITY_PROMPT = (
    "Produce ONE self-contained textual description that fully encodes this "
    "page. The page image is NOT available downstream - only your text stands "
    "in for it. Transcribe every heading, label, value, dose, date, and table "
    "cell you can read. Describe the structure and the relationships between "
    "elements - reading order, sequence, decision pathways (if X then Y / go "
    "to page N), dependencies, and groupings - so the page's logic is fully "
    "preserved. Use Markdown: tables for tabular data, headings to mirror the "
    "page. Be exhaustive and specific, but do not infer beyond what is shown; "
    "where the page is genuinely unclear, say so rather than guess.\n\n"
    "Return a JSON object with two fields: \"title\" (a short factual label of "
    "what this page covers) and \"prose\" (the full Markdown description "
    "described above)."
)

_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "prose": {"type": "string"}},
    "required": ["title", "prose"],
    "additionalProperties": False,
}


def build_messages(image_bytes: bytes, raw_text: str) -> list[dict]:
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    text = (
        FIDELITY_PROMPT
        + "\n\nExact text extracted from this page (ground truth for values):\n"
        + raw_text
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                },
                {"type": "text", "text": text},
            ],
        }
    ]


def parse_description(text: str) -> tuple[str, str]:
    data = json.loads(text)
    if "title" not in data or "prose" not in data:
        raise ValueError(f"response missing title/prose: {sorted(data)}")
    return data["title"], data["prose"]


def describe_page(
    client,
    image_bytes: bytes,
    raw_text: str,
    model: str = "claude-opus-4-8",
    max_tokens: int = 16000,
) -> tuple[str, str]:
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=build_messages(image_bytes, raw_text),
    ) as stream:
        message = stream.get_final_message()
    text = next(b.text for b in message.content if b.type == "text")
    return parse_description(text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_describe.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/guideline_extractor/describe.py tests/test_describe.py
git commit -m "feat: fidelity-contract page description via Claude Opus 4.8"
```

---

### Task 5: On-disk storage layout

**Files:**
- Create: `src/guideline_extractor/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: `PageRecord`, `Manifest` from `models` (Task 1).
- Produces:
  - `image_filename(pdf_index: int) -> str` → `"pages/pNNN.png"` (zero-padded to 3, wider if needed).
  - `record_filename(pdf_index: int) -> str` → `"pages/pNNN.json"`.
  - `save_image(base_dir: str, pdf_index: int, image_bytes: bytes) -> str` — writes the PNG, returns its relative path.
  - `write_page_record(base_dir: str, record: PageRecord) -> None` — writes `pages/pNNN.json` (by `pdf_index`).
  - `write_manifest(base_dir: str, manifest: Manifest) -> None` — writes `manifest.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
import json

from guideline_extractor.models import Manifest, PageMapEntry, PageRecord
from guideline_extractor.storage import (
    image_filename,
    record_filename,
    save_image,
    write_page_record,
    write_manifest,
)


def test_filenames_zero_pad_to_three():
    assert image_filename(7) == "pages/p007.png"
    assert record_filename(40) == "pages/p040.json"


def test_save_image_writes_bytes_and_returns_relative_path(tmp_path):
    rel = save_image(str(tmp_path), 3, b"\x89PNG")
    assert rel == "pages/p003.png"
    assert (tmp_path / "pages" / "p003.png").read_bytes() == b"\x89PNG"


def test_write_page_record_serializes_by_pdf_index(tmp_path):
    rec = PageRecord("g", 38, 40, "T", "P", "pages/p040.png", "raw")
    write_page_record(str(tmp_path), rec)
    written = json.loads((tmp_path / "pages" / "p040.json").read_text())
    assert written["page_number"] == 38
    assert written["pdf_index"] == 40


def test_write_manifest_writes_json(tmp_path):
    m = Manifest("g", "T", "ZA", None, "2023", None, "a.pdf", 1,
                 [PageMapEntry(1, "Preface", 1)])
    write_manifest(str(tmp_path), m)
    written = json.loads((tmp_path / "manifest.json").read_text())
    assert written["guideline_id"] == "g"
    assert written["pages"][0]["title"] == "Preface"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guideline_extractor.storage'`

- [ ] **Step 3: Write the implementation**

```python
# src/guideline_extractor/storage.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/guideline_extractor/storage.py tests/test_storage.py
git commit -m "feat: on-disk storage layout for records and manifest"
```

---

### Task 6: Pipeline orchestration + CLI

**Files:**
- Create: `src/guideline_extractor/pipeline.py`
- Create: `src/guideline_extractor/cli.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `render_pdf`/`RenderedPage` (Task 3), `detect_printed_number`/`calibrate_offset`/`assign_page_numbers`/`check_monotonic` (Task 2), `describe_page` (Task 4), `save_image`/`image_filename`/`write_page_record`/`write_manifest` (Task 5), `PageRecord`/`PageMapEntry`/`Manifest` (Task 1).
- Produces:
  - `extract(pdf_path: str, out_dir: str, guideline_id: str, *, guideline_title: str, jurisdiction: str | None = None, publisher: str | None = None, version: str | None = None, effective_date: str | None = None, client=None, describe_fn=describe_page, rendered=None) -> tuple[Manifest, list[int]]` — runs the full pipeline, writes all files, returns the manifest and the QC flag list (0-based indices of pages whose printed numbering broke). `client`/`describe_fn`/`rendered` are injectable for testing.
  - `main(argv: list[str] | None = None) -> int` in `cli.py`.

- [ ] **Step 1: Write the failing test**

The test injects pre-built `rendered` pages and a `describe_fn` stub so no PDF or network is touched.

```python
# tests/test_pipeline.py
import json

from guideline_extractor.render import RenderedPage
from guideline_extractor.pipeline import extract


def _fake_describe(client, image_bytes, raw_text):
    # title echoes the raw text's first line so assertions are meaningful
    return raw_text.splitlines()[0], f"prose for {raw_text.splitlines()[0]}"


def test_extract_writes_manifest_and_one_record_per_page(tmp_path):
    rendered = [
        RenderedPage(1, b"\x89PNG1", "Cough\n1"),
        RenderedPage(2, b"\x89PNG2", "TB treatment\n2"),
    ]
    manifest, flags = extract(
        "ignored.pdf",
        str(tmp_path),
        "APC_2023_ZA",
        guideline_title="Adult Primary Care 2023",
        jurisdiction="South Africa",
        describe_fn=_fake_describe,
        rendered=rendered,
    )
    assert manifest.page_count == 2
    assert flags == []
    # page numbers calibrated: printed 1,2 on sheets 1,2 -> offset 0
    assert [p.page_number for p in manifest.pages] == [1, 2]
    assert manifest.pages[1].title == "TB treatment"

    rec = json.loads((tmp_path / "pages" / "p002.json").read_text())
    assert rec["page_number"] == 2
    assert rec["title"] == "TB treatment"
    assert rec["image_path"] == "pages/p002.png"
    assert (tmp_path / "pages" / "p002.png").read_bytes() == b"\x89PNG2"
    assert (tmp_path / "manifest.json").exists()


def test_extract_flags_broken_numbering(tmp_path):
    # printed numbers 5 then 3 -> not strictly increasing -> flag index 1
    rendered = [
        RenderedPage(1, b"a", "A\n5"),
        RenderedPage(2, b"b", "B\n3"),
    ]
    _, flags = extract(
        "ignored.pdf", str(tmp_path), "g",
        guideline_title="t", describe_fn=_fake_describe, rendered=rendered,
    )
    assert flags == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guideline_extractor.pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# src/guideline_extractor/pipeline.py
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
from .storage import image_filename, save_image, write_manifest, write_page_record


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

    # Calibrate printed page numbers against sheet indices, then QC.
    samples = [(p.pdf_index, detect_printed_number(p.raw_text)) for p in pages]
    offset = calibrate_offset(samples)
    page_numbers = assign_page_numbers([p.pdf_index for p in pages], offset)
    flags = check_monotonic(page_numbers)

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
```

```python
# src/guideline_extractor/cli.py
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
        print(f"WARNING: page-number QC flagged sheet indices (0-based): {flags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v && pytest -q`
Expected: `test_pipeline.py` PASS (2 tests); full suite PASS

- [ ] **Step 5: Commit**

```bash
git add src/guideline_extractor/pipeline.py src/guideline_extractor/cli.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestration and CLI entrypoint"
```

---

## Manual verification (after all tasks)

Run the real pipeline against the APC guideline and spot-check prose quality on a flowchart page:

```bash
python -m guideline_extractor.cli \
  ~/Downloads/APC_2023_Clinical_tool-EBOOK.pdf \
  ./guidelines/APC_2023_ZA \
  --guideline-id APC_2023_ZA \
  --guideline-title "Adult Primary Care (APC) 2023" \
  --jurisdiction "South Africa" \
  --version 2023
```

Then open a known flowchart page's `pages/pNNN.json`, read its `prose`, and compare against the `.png` — confirm decision pathways survived as `if X → then Y / go to p.N`, tables render as Markdown tables, and any QC-flagged pages are inspected against their images.

## Notes / deviations from the spec

- **Filenames use `pdf_index`, not `page_number`** (spec §4.1 showed `page_number`-named JSON). Rationale: `pdf_index` is always unique and calibration-independent, so the filesystem never collides even when printed numbering is broken/duplicated. The manifest `pages` map carries both `page_number` and `pdf_index`, so the agent still resolves a numeric reference → sheet in one lookup. Flag for the reviewer if strict spec alignment is preferred.
- Retrieval (`get_page` / `search_prose`), embeddings, and the querying agent are **out of scope** here — the spec (§6) documents them as downstream; this plan builds only the extraction half whose output supports them.
