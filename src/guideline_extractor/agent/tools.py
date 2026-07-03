import re

from rank_bm25 import BM25Plus

from .. import library

_INDEX_CACHE: dict[str, tuple] = {}  # guideline_id -> (BM25Plus, list[pageinfo])
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
        bm25 = BM25Plus([_tokenize(p["prose"]) for p in pages])
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
