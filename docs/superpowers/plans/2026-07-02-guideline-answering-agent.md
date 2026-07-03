# Guideline Answering Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A grounded answering agent that takes a clinical query, navigates the extracted guidelines with tools, and returns an answer cited to specific pages — or declines when the guidelines don't cover it.

**Architecture:** A read-only `library` layer over `guidelines/`; a `tools` layer (list/search/get, BM25 over prose) built on it; an OpenAI tool-calling `loop` that reasons and terminates by calling `submit_answer`; and two entrypoints (CLI + a web `/api/ask` route with a query box). Retrieval is embeddings-free (BM25 in-memory).

**Tech Stack:** Python 3.11+, `openai` (chat completions tool calling), `rank-bm25`, `fastapi` (existing UI), `pytest`.

## Global Constraints

- Reads the extraction output under `GE_OUTPUT_ROOT` (default `./guidelines`), layout `guidelines/<id>/manifest.json` + `pages/pNNN.json` (files named by `pdf_index`).
- Page records: `{guideline_id, page_number, pdf_index, title, prose, image_path, raw_text}`. Manifest: `{guideline_id, title, jurisdiction, publisher, version, effective_date, source_file, page_count, pages:[{page_number, title, pdf_index}]}`.
- Agent model resolves `model` arg → `OPENAI_MODEL` env → `"gpt-5.5"`.
- The agent answers ONLY from tool output; it must cite pages or decline. Grounding lives in the system prompt.
- `submit_answer(answer, citations)` is the terminal tool; `citations = [{guideline_id, page_number}]`.
- No network in tests: the loop takes an injectable `client`; tools/library read a fixture `guidelines/` dir via `GE_OUTPUT_ROOT`.
- Path safety: reject `guideline_id` values that escape the root.
- Traceability: the loop emits `on_event` `tool_call`/`tool_result` events as it navigates; the CLI prints them and `/api/ask` streams NDJSON (`tool_call`/`tool_result` → terminal `done`/`error`) so the UI shows the agent navigating live.

---

### Task 1: Library — read-only access to `guidelines/`

**Files:**
- Create: `src/guideline_extractor/library.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Produces:
  - `root() -> pathlib.Path` — `GE_OUTPUT_ROOT` (default `guidelines`), resolved.
  - `list_guidelines() -> list[dict]` — `[{guideline_id, title, jurisdiction, page_count}]`, sorted by id; `[]` if root missing.
  - `load_manifest(guideline_id: str) -> dict` — raises `LookupError` if unknown/invalid id.
  - `load_page(guideline_id: str, page_number: int) -> dict | None` — the page record whose `page_number` matches, or `None`.
  - `neighbors(guideline_id: str, page_number: int) -> tuple[dict | None, dict | None]` — `(prev, next)` each `{page_number, title}` or `None`, ordered by `page_number`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_library.py
import json
import pytest
from guideline_extractor import library


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path))
    d = tmp_path / "APC"
    (d / "pages").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "guideline_id": "APC", "title": "APC 2023", "jurisdiction": "South Africa",
        "publisher": None, "version": "2023", "effective_date": None,
        "source_file": "a.pdf", "page_count": 2,
        "pages": [
            {"page_number": 10, "title": "Cough", "pdf_index": 1},
            {"page_number": 11, "title": "TB", "pdf_index": 2},
        ],
    }))
    (d / "pages" / "p001.json").write_text(json.dumps({
        "guideline_id": "APC", "page_number": 10, "pdf_index": 1, "title": "Cough",
        "prose": "cough prose", "image_path": "pages/p001.png", "raw_text": "raw1"}))
    (d / "pages" / "p002.json").write_text(json.dumps({
        "guideline_id": "APC", "page_number": 11, "pdf_index": 2, "title": "TB",
        "prose": "tb prose", "image_path": "pages/p002.png", "raw_text": "raw2"}))
    return tmp_path


def test_list_guidelines(lib):
    assert library.list_guidelines() == [
        {"guideline_id": "APC", "title": "APC 2023", "jurisdiction": "South Africa", "page_count": 2}
    ]


def test_list_guidelines_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path / "nope"))
    assert library.list_guidelines() == []


def test_load_page_by_printed_number(lib):
    rec = library.load_page("APC", 11)
    assert rec["pdf_index"] == 2
    assert rec["prose"] == "tb prose"


def test_load_page_missing_returns_none(lib):
    assert library.load_page("APC", 999) is None


def test_load_manifest_unknown_raises(lib):
    with pytest.raises(LookupError):
        library.load_manifest("../etc")


def test_neighbors(lib):
    prev, nxt = library.neighbors("APC", 10)
    assert prev is None
    assert nxt == {"page_number": 11, "title": "TB"}
    prev, nxt = library.neighbors("APC", 11)
    assert prev == {"page_number": 10, "title": "Cough"}
    assert nxt is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_library.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'guideline_extractor.library'`

- [ ] **Step 3: Write the implementation**

```python
# src/guideline_extractor/library.py
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
    for child in sorted(r.iterdir()):
        m = child / "manifest.json"
        if m.is_file():
            d = json.loads(m.read_text())
            out.append({
                "guideline_id": d["guideline_id"],
                "title": d["title"],
                "jurisdiction": d.get("jurisdiction"),
                "page_count": d["page_count"],
            })
    return out


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_library.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/guideline_extractor/library.py tests/test_library.py
git commit -m "feat: read-only library over guidelines output"
```

---

### Task 2: Tools — list/search/get + BM25, and the OpenAI tool specs

**Files:**
- Modify: `pyproject.toml` (add `rank-bm25` to base deps)
- Create: `src/guideline_extractor/agent/__init__.py`
- Create: `src/guideline_extractor/agent/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `library.list_guidelines`, `library.load_manifest`, `library.load_page`, `library.neighbors` (Task 1).
- Produces:
  - `search_pages(guideline_id: str, query: str, k: int = 8) -> list[dict]` — ranked `[{page_number, title, snippet}]` (BM25 over prose; only positive scores; empty list if no matches).
  - `get_page(guideline_id: str, page_number: int) -> dict` — `{page_number, title, prose, prev, next}` or `{"error": str}`.
  - `list_guidelines() -> list[dict]` — delegates to `library.list_guidelines`.
  - `run_read_tool(name: str, args: dict) -> dict | list` — dispatch for `list_guidelines`/`search_pages`/`get_page`; returns `{"error": ...}` on bad input.
  - `TOOL_SPECS: list[dict]` — OpenAI function-tool schemas for all four tools including `submit_answer`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools.py
import json
import pytest
from guideline_extractor.agent import tools


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path))
    tools._INDEX_CACHE.clear()
    d = tmp_path / "APC"
    (d / "pages").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "guideline_id": "APC", "title": "APC 2023", "jurisdiction": "South Africa",
        "publisher": None, "version": "2023", "effective_date": None,
        "source_file": "a.pdf", "page_count": 2,
        "pages": [
            {"page_number": 10, "title": "Cough", "pdf_index": 1},
            {"page_number": 11, "title": "TB treatment", "pdf_index": 2},
        ],
    }))
    (d / "pages" / "p001.json").write_text(json.dumps({
        "guideline_id": "APC", "page_number": 10, "pdf_index": 1, "title": "Cough",
        "prose": "Assess cough. Night sweats and weight loss suggest TB screening.",
        "image_path": "pages/p001.png", "raw_text": "r"}))
    (d / "pages" / "p002.json").write_text(json.dumps({
        "guideline_id": "APC", "page_number": 11, "pdf_index": 2, "title": "TB treatment",
        "prose": "Start the rifampicin isoniazid regimen once Xpert is positive.",
        "image_path": "pages/p002.png", "raw_text": "r"}))
    return tmp_path


def test_search_pages_ranks_body_content(lib):
    hits = tools.search_pages("APC", "night sweats weight loss")
    assert hits[0]["page_number"] == 10
    assert "title" in hits[0] and "snippet" in hits[0]


def test_search_pages_finds_treatment(lib):
    hits = tools.search_pages("APC", "rifampicin regimen")
    assert hits[0]["page_number"] == 11


def test_search_pages_no_match_is_empty(lib):
    assert tools.search_pages("APC", "cardiology echocardiogram angioplasty") == []


def test_get_page_returns_prose_and_neighbors(lib):
    p = tools.get_page("APC", 10)
    assert p["prose"].startswith("Assess cough")
    assert p["prev"] is None
    assert p["next"] == {"page_number": 11, "title": "TB treatment"}


def test_get_page_unknown_is_error(lib):
    assert "error" in tools.get_page("APC", 999)


def test_run_read_tool_dispatch(lib):
    assert tools.run_read_tool("list_guidelines", {})[0]["guideline_id"] == "APC"
    assert "error" in tools.run_read_tool("nope", {})


def test_tool_specs_cover_all_four():
    names = {t["function"]["name"] for t in tools.TOOL_SPECS}
    assert names == {"list_guidelines", "search_pages", "get_page", "submit_answer"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tools.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'guideline_extractor.agent'`

- [ ] **Step 3: Add the dependency, then implement**

In `pyproject.toml`, add `rank-bm25` to the base dependencies list:

```toml
dependencies = ["pymupdf>=1.24", "openai>=1.40", "python-dotenv>=1.0", "rank-bm25>=0.2"]
```

Reinstall: `.venv/bin/pip install -e ".[dev]"`

```python
# src/guideline_extractor/agent/__init__.py
```

```python
# src/guideline_extractor/agent/tools.py
import re

from rank_bm25 import BM25Okapi

from .. import library

_INDEX_CACHE: dict[str, tuple] = {}  # guideline_id -> (BM25Okapi, list[pageinfo])
_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _index(guideline_id: str):
    if guideline_id not in _INDEX_CACHE:
        pages = []
        for entry in library.load_manifest(guideline_id)["pages"]:
            rec = library.load_page(guideline_id, entry["page_number"])
            prose = rec["prose"] if rec else ""
            pages.append({"page_number": entry["page_number"], "title": entry["title"], "prose": prose})
        bm25 = BM25Okapi([_tokenize(p["prose"]) for p in pages])
        _INDEX_CACHE[guideline_id] = (bm25, pages)
    return _INDEX_CACHE[guideline_id]


def _snippet(prose: str, query: str, width: int = 200) -> str:
    low = prose.lower()
    for term in _tokenize(query):
        i = low.find(term)
        if i >= 0:
            start = max(0, i - width // 2)
            return prose[start:start + width].strip()
    return prose[:width].strip()


def list_guidelines() -> list[dict]:
    return library.list_guidelines()


def search_pages(guideline_id: str, query: str, k: int = 8) -> list[dict]:
    bm25, pages = _index(guideline_id)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(pages)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in ranked[:k]:
        if scores[i] <= 0:
            break
        p = pages[i]
        out.append({"page_number": p["page_number"], "title": p["title"],
                    "snippet": _snippet(p["prose"], query)})
    return out


def get_page(guideline_id: str, page_number: int) -> dict:
    rec = library.load_page(guideline_id, page_number)
    if rec is None:
        return {"error": f"no page {page_number} in {guideline_id}"}
    prev, nxt = library.neighbors(guideline_id, page_number)
    return {"page_number": rec["page_number"], "title": rec["title"],
            "prose": rec["prose"], "prev": prev, "next": nxt}


def run_read_tool(name: str, args: dict):
    try:
        if name == "list_guidelines":
            return list_guidelines()
        if name == "search_pages":
            return search_pages(args["guideline_id"], args["query"], int(args.get("k", 8)))
        if name == "get_page":
            return get_page(args["guideline_id"], int(args["page_number"]))
        return {"error": f"unknown tool: {name}"}
    except LookupError as exc:
        return {"error": str(exc)}
    except (KeyError, ValueError, TypeError) as exc:
        return {"error": f"bad arguments for {name}: {exc}"}


TOOL_SPECS: list[dict] = [
    {"type": "function", "function": {
        "name": "list_guidelines",
        "description": "List the guidelines available to answer from, with title and jurisdiction.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "search_pages",
        "description": "Search a guideline's page prose for a query; returns ranked pages with snippets.",
        "parameters": {"type": "object", "properties": {
            "guideline_id": {"type": "string"},
            "query": {"type": "string"},
            "k": {"type": "integer"}},
            "required": ["guideline_id", "query"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_page",
        "description": "Read the full prose of one page plus its neighboring pages.",
        "parameters": {"type": "object", "properties": {
            "guideline_id": {"type": "string"},
            "page_number": {"type": "integer"}},
            "required": ["guideline_id", "page_number"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the final grounded answer and the pages it is based on. Ends the task.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "object", "properties": {
                "guideline_id": {"type": "string"},
                "page_number": {"type": "integer"}},
                "required": ["guideline_id", "page_number"], "additionalProperties": False}}},
            "required": ["answer", "citations"], "additionalProperties": False}}},
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tools.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/guideline_extractor/agent/__init__.py src/guideline_extractor/agent/tools.py tests/test_tools.py
git commit -m "feat: agent tools (list/search/get) with BM25 and OpenAI tool specs"
```

---

### Task 3: The agent loop

**Files:**
- Create: `src/guideline_extractor/agent/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `tools.TOOL_SPECS`, `tools.run_read_tool` (Task 2); `library.load_page` (Task 1).
- Produces:
  - `AnswerResult` dataclass: `answer: str`, `citations: list[dict]` (`[{guideline_id, page_number, title}]`), `complete: bool`.
  - `SYSTEM_PROMPT: str`.
  - `answer(query: str, client=None, model: str | None = None, max_turns: int = 12, on_event=None) -> AnswerResult`.
    `on_event`, if given, is called with `{"type": "tool_call", "name", "args"}` before each tool runs
    and `{"type": "tool_result", "name", "summary"}` after (short human summary). `submit_answer` is
    not emitted as an event.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loop.py
import json
import pytest
from guideline_extractor.agent import loop
from guideline_extractor.agent import tools


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path))
    tools._INDEX_CACHE.clear()
    d = tmp_path / "APC"
    (d / "pages").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "guideline_id": "APC", "title": "APC 2023", "jurisdiction": "South Africa",
        "publisher": None, "version": "2023", "effective_date": None,
        "source_file": "a.pdf", "page_count": 1,
        "pages": [{"page_number": 10, "title": "Cough", "pdf_index": 1}]}))
    (d / "pages" / "p001.json").write_text(json.dumps({
        "guideline_id": "APC", "page_number": 10, "pdf_index": 1, "title": "Cough",
        "prose": "Cough >= 2 weeks: screen for TB.", "image_path": "x", "raw_text": "r"}))
    return tmp_path


# --- fake OpenAI client: scripted list of responses ---
def _tool_call(cid, name, args):
    fn = type("Fn", (), {"name": name, "arguments": json.dumps(args)})()
    return type("TC", (), {"id": cid, "function": fn})()


def _response(content=None, tool_calls=None):
    msg = type("Msg", (), {"content": content, "tool_calls": tool_calls})()
    choice = type("Choice", (), {"message": msg})()
    return type("Resp", (), {"choices": [choice]})()


class _FakeChat:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._scripted.pop(0)


class _FakeClient:
    def __init__(self, scripted):
        self.chat = type("C", (), {"completions": _FakeChat(scripted)})()


def test_answer_runs_tools_then_submits(lib):
    client = _FakeClient([
        _response(tool_calls=[_tool_call("1", "search_pages", {"guideline_id": "APC", "query": "cough"})]),
        _response(tool_calls=[_tool_call("2", "get_page", {"guideline_id": "APC", "page_number": 10})]),
        _response(tool_calls=[_tool_call("3", "submit_answer", {
            "answer": "Screen for TB if cough >= 2 weeks.",
            "citations": [{"guideline_id": "APC", "page_number": 10}]})]),
    ])
    result = loop.answer("what to do for a long cough", client=client)
    assert result.complete is True
    assert "TB" in result.answer
    assert result.citations == [{"guideline_id": "APC", "page_number": 10, "title": "Cough"}]
    # the get_page result was fed back as a tool message before the final turn
    last_msgs = client.chat.completions.calls[-1]["messages"]
    assert any(m.get("role") == "tool" for m in last_msgs)


def test_answer_emits_trace_events(lib):
    client = _FakeClient([
        _response(tool_calls=[_tool_call("1", "search_pages", {"guideline_id": "APC", "query": "cough"})]),
        _response(tool_calls=[_tool_call("2", "get_page", {"guideline_id": "APC", "page_number": 10})]),
        _response(tool_calls=[_tool_call("3", "submit_answer", {"answer": "a", "citations": []})]),
    ])
    events = []
    loop.answer("q", client=client, on_event=events.append)
    kinds = [(e["type"], e["name"]) for e in events]
    assert ("tool_call", "search_pages") in kinds
    assert ("tool_result", "search_pages") in kinds
    assert ("tool_call", "get_page") in kinds
    # submit_answer is not emitted as a tool event
    assert all(e["name"] != "submit_answer" for e in events)
    # results carry a short human summary string
    assert all(isinstance(e["summary"], str) for e in events if e["type"] == "tool_result")


def test_answer_can_decline(lib):
    client = _FakeClient([
        _response(tool_calls=[_tool_call("1", "submit_answer", {
            "answer": "The guidelines do not cover this.", "citations": []})]),
    ])
    result = loop.answer("what is the capital of France", client=client)
    assert result.complete is True
    assert result.citations == []


def test_answer_incomplete_when_turns_exhausted(lib):
    # always asks to search, never submits
    scripted = [_response(tool_calls=[_tool_call(str(i), "search_pages",
                {"guideline_id": "APC", "query": "x"})]) for i in range(5)]
    client = _FakeClient(scripted)
    result = loop.answer("q", client=client, max_turns=3)
    assert result.complete is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'guideline_extractor.agent.loop'`

- [ ] **Step 3: Write the implementation**

```python
# src/guideline_extractor/agent/loop.py
import json
import os
from dataclasses import dataclass, field

from .. import library
from . import tools

SYSTEM_PROMPT = (
    "You answer clinical questions using ONLY the provided guideline tools. Do not use outside "
    "medical knowledge and do not infer beyond what the pages say.\n"
    "Workflow: call list_guidelines to see what is available and pick the relevant one(s) by "
    "jurisdiction and title. Use search_pages to find an entry page, get_page to read it, and open "
    "neighbouring pages (via get_page on the prev/next page numbers) when a flowchart, table, or the "
    "answer itself continues.\n"
    "Cite every page you used. When done, call submit_answer with the answer and its citations. "
    "If the guidelines do not address the question, or no guideline is relevant, call submit_answer "
    "saying so plainly - do not guess."
)


@dataclass
class AnswerResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    complete: bool = False


def _enrich(citations) -> list[dict]:
    out = []
    for c in citations or []:
        gid, pn = c.get("guideline_id"), c.get("page_number")
        if gid is None or pn is None:
            continue
        title = None
        try:
            rec = library.load_page(gid, int(pn))
            title = rec["title"] if rec else None
        except LookupError:
            title = None
        out.append({"guideline_id": gid, "page_number": pn, "title": title})
    return out


def _assistant_message(msg) -> dict:
    return {
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ],
    }


def _summarize(name: str, result) -> str:
    if isinstance(result, dict) and "error" in result:
        return result["error"]
    if name == "search_pages":
        if not result:
            return "no matches"
        return f"{len(result)} pages: " + ", ".join(f"p.{h['page_number']}" for h in result)
    if name == "get_page":
        return f"p.{result['page_number']} — {result.get('title', '')}"
    if name == "list_guidelines":
        return f"{len(result)} guidelines"
    return ""


def answer(query: str, client=None, model: str | None = None, max_turns: int = 12,
           on_event=None) -> AnswerResult:
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    model = model or os.environ.get("OPENAI_MODEL", "gpt-5.5")

    def emit(event):
        if on_event is not None:
            on_event(event)

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}]

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools.TOOL_SPECS, tool_choice="auto")
        msg = response.choices[0].message
        if not msg.tool_calls:
            return AnswerResult(msg.content or "", [], False)

        messages.append(_assistant_message(msg))
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            if tc.function.name == "submit_answer":
                return AnswerResult(args.get("answer", ""), _enrich(args.get("citations")), True)
            emit({"type": "tool_call", "name": tc.function.name, "args": args})
            result = tools.run_read_tool(tc.function.name, args)
            emit({"type": "tool_result", "name": tc.function.name,
                  "summary": _summarize(tc.function.name, result)})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    return AnswerResult("", [], False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_loop.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/guideline_extractor/agent/loop.py tests/test_loop.py
git commit -m "feat: grounded answering agent loop with submit_answer terminal tool"
```

---

### Task 4: CLI entrypoint

**Files:**
- Create: `src/guideline_extractor/ask.py`
- Test: `tests/test_ask.py`

**Interfaces:**
- Consumes: `loop.answer` (Task 3), `AnswerResult`.
- Produces: `main(argv: list[str] | None = None) -> int` — prints the answer and a `Sources:` list; loads `.env`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ask.py
from guideline_extractor import ask
from guideline_extractor.agent.loop import AnswerResult


def test_ask_prints_answer_and_sources(capsys, monkeypatch):
    monkeypatch.setattr(ask, "answer", lambda q, **k: AnswerResult(
        "Screen for TB.", [{"guideline_id": "APC", "page_number": 10, "title": "Cough"}], True))
    rc = ask.main(["what to do for a long cough"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Screen for TB." in out
    assert "APC" in out and "p.10" in out and "Cough" in out


def test_ask_prints_live_trace(capsys, monkeypatch):
    def fake_answer(q, on_event=None, **k):
        if on_event:
            on_event({"type": "tool_call", "name": "search_pages", "args": {"query": "cough"}})
            on_event({"type": "tool_result", "name": "search_pages", "summary": "1 pages: p.10"})
        return AnswerResult("ok", [], True)
    monkeypatch.setattr(ask, "answer", fake_answer)
    ask.main(["q"])
    err = capsys.readouterr().err
    assert "search_pages" in err


def test_ask_notes_incomplete(capsys, monkeypatch):
    monkeypatch.setattr(ask, "answer", lambda q, **k: AnswerResult("", [], False))
    rc = ask.main(["q"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no answer" in out.lower() or "incomplete" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ask.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'guideline_extractor.ask'`

- [ ] **Step 3: Write the implementation**

```python
# src/guideline_extractor/ask.py
import argparse
import sys

from dotenv import load_dotenv

from .agent.loop import answer

load_dotenv()


def _print_step(event: dict) -> None:
    if event["type"] == "tool_call":
        print(f"  → {event['name']}({event.get('args', {})})", file=sys.stderr)
    elif event["type"] == "tool_result":
        print(f"    {event['name']}: {event['summary']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question over the extracted guidelines.")
    parser.add_argument("query")
    args = parser.parse_args(argv)

    result = answer(args.query, on_event=_print_step)
    if not result.complete or not result.answer:
        print("No answer (the agent could not complete or found nothing relevant).")
        return 0

    print(result.answer)
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            title = f" — {c['title']}" if c.get("title") else ""
            print(f"  {c['guideline_id']} p.{c['page_number']}{title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ask.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/guideline_extractor/ask.py tests/test_ask.py
git commit -m "feat: CLI to ask the guideline agent"
```

---

### Task 5: Web `/api/ask` + query box in the UI

**Files:**
- Modify: `src/guideline_extractor/webapp.py`
- Test: `tests/test_webapp_ask.py`

**Interfaces:**
- Consumes: `loop.answer` (Task 3) — including its `on_event` callback.
- Produces:
  - `POST /api/ask` — body `{"query": str}` → **streams NDJSON**: `tool_call` / `tool_result` events
    (from `on_event`) as the agent navigates, then a terminal `{"type": "done", "answer", "citations",
    "complete"}` or `{"type": "error", "detail"}`.
  - A query box + live-trace/answer panel in `INDEX_HTML`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_webapp_ask.py
import json
from fastapi.testclient import TestClient
from guideline_extractor import webapp
from guideline_extractor.agent.loop import AnswerResult


def test_ask_endpoint_streams_trace_then_done(monkeypatch, tmp_path):
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path))

    def fake_answer(q, on_event=None, **k):
        if on_event:
            on_event({"type": "tool_call", "name": "search_pages", "args": {"query": "cough"}})
            on_event({"type": "tool_result", "name": "search_pages", "summary": "1 pages: p.10"})
        return AnswerResult("Screen for TB.",
                            [{"guideline_id": "APC", "page_number": 10, "title": "Cough"}], True)

    monkeypatch.setattr(webapp, "answer", fake_answer)
    resp = TestClient(webapp.app).post("/api/ask", json={"query": "long cough"})
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert any(e["type"] == "tool_call" and e["name"] == "search_pages" for e in events)
    assert any(e["type"] == "tool_result" for e in events)
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "Screen for TB."
    assert events[-1]["citations"][0]["page_number"] == 10
    assert events[-1]["complete"] is True


def test_ask_endpoint_streams_error(monkeypatch, tmp_path):
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path))
    def boom(q, **k):
        raise RuntimeError("no api key")
    monkeypatch.setattr(webapp, "answer", boom)
    resp = TestClient(webapp.app).post("/api/ask", json={"query": "x"})
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert events[-1]["type"] == "error"
    assert "no api key" in events[-1]["detail"]


def test_index_has_ask_box(monkeypatch, tmp_path):
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path))
    html = TestClient(webapp.app).get("/").text
    assert 'id="ask"' in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_webapp_ask.py -q`
Expected: FAIL — `AttributeError: module 'guideline_extractor.webapp' has no attribute 'answer'` (or 404 on `/api/ask`)

- [ ] **Step 3: Write the implementation**

Add the import near the top of `webapp.py` (with the other local imports):

```python
from .agent.loop import answer
```

`webapp.py` already imports `json`, `queue`, `threading`, `StreamingResponse`, and `HTTPException`
(from the extraction streaming endpoint). Add the endpoint (place it near the other `/api` routes,
e.g. before the `index()` handler):

```python
@app.post("/api/ask")
def run_ask(body: dict) -> StreamingResponse:
    query = (body or {}).get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="empty query")

    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            result = answer(query, on_event=events.put)
            events.put({"type": "done", "answer": result.answer,
                        "citations": result.citations, "complete": result.complete})
        except Exception as exc:  # surface auth/model errors as a stream event
            events.put({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            item = events.get()
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
```

Note: `answer`'s `on_event` receives `tool_call`/`tool_result` dicts, which are JSON-serializable, so
`events.put` can be passed directly as the callback.

In `INDEX_HTML`, add an ask bar just inside `<main>` — replace the opening of the `<main>` block:

Find:
```html
<main>
  <div id="list"></div>
  <div id="block"><div class="empty">Select a page.</div></div>
</main>
```
Replace with:
```html
<main>
  <div id="list"></div>
  <div id="block">
    <div id="askbar" style="padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:8px;">
      <input type="text" id="ask" placeholder="Ask a question over the guidelines" style="flex:1;padding:5px 8px;border:1px solid var(--line);font:13px inherit;">
      <button id="askbtn" style="padding:5px 12px;cursor:pointer;">Ask</button>
    </div>
    <div id="answer" class="pane on" style="padding:16px;"><div class="empty">Ask a question, or select a page.</div></div>
  </div>
</main>
```

Add this handler to the `<script>` (before the final `loadGuidelines();` call). It reads the NDJSON
stream and shows the trace live, then the answer:

```javascript
async function runAsk() {
  const q = $('#ask').value.trim();
  if (!q) return;
  const trace = [];
  const render = (answerHtml) => {
    const traceHtml = trace.length
      ? '<div style="color:var(--muted);font-size:12px;font-family:ui-monospace,Menlo,monospace;margin-bottom:12px;">'
        + trace.map(escapeHtml).join('<br>') + '</div>'
      : '';
    $('#answer').innerHTML = traceHtml + (answerHtml || '');
  };
  trace.push('thinking...');
  render('');
  const res = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query:q})});
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream:true});
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const m = JSON.parse(line);
      if (m.type === 'tool_call') {
        if (trace[trace.length-1] === 'thinking...') trace.pop();
        trace.push('→ ' + m.name + '(' + JSON.stringify(m.args) + ')');
        render('');
      } else if (m.type === 'tool_result') {
        trace.push('   ' + m.name + ': ' + m.summary);
        render('');
      } else if (m.type === 'done') {
        let html = '<div class="prose">' + escapeHtml(m.answer || '(no answer)') + '</div>';
        if (m.citations && m.citations.length) {
          html += '<p style="color:var(--muted);font-size:12px;">Sources: '
            + m.citations.map(c => escapeHtml(c.guideline_id)+' p.'+c.page_number+(c.title?(' — '+escapeHtml(c.title)):'')).join('; ')
            + '</p>';
        }
        render(html);
      } else if (m.type === 'error') {
        render('<div class="empty">error: ' + escapeHtml(m.detail) + '</div>');
      }
    }
  }
}
$('#askbtn').onclick = runAsk;
$('#ask').onkeydown = e => { if (e.key === 'Enter') runAsk(); };
```

Note: `selectPage` overwrites `#block`'s innerHTML; that is acceptable — clicking a page replaces the
answer view with the page block, and re-running a query is done from the ask bar which reappears on
reload. Leave that behavior as-is (YAGNI).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_webapp_ask.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the whole suite and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all tests)

```bash
git add src/guideline_extractor/webapp.py tests/test_webapp_ask.py
git commit -m "feat: /api/ask endpoint and query box in the UI"
```

---

## Manual verification (after all tasks)

With a real key in `.env` and an extraction present (e.g. `guidelines/1`):

```bash
.venv/bin/python -m guideline_extractor.ask "what should I do for an adult with a cough for 3 weeks and weight loss?"
```

Confirm the answer is grounded (mentions TB screening per the guideline) and lists Sources with page numbers. Then start the web app, type the same question in the ask bar, and confirm the answer + sources render.

## Notes
- Embeddings, cross-guideline fan-out/reduce, and multi-turn memory are out of scope (see spec §10). The tool signatures are the seam to add embeddings later.
- `library.py` duplicates the small root/read logic the webapp does inline; left as-is to stay focused (spec §7).
