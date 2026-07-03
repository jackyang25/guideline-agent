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
        except (LookupError, ValueError, TypeError):
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
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                if tc.function.name == "submit_answer":
                    return AnswerResult("", [], False)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                  "content": json.dumps({"error": "invalid tool arguments"})})
                continue
            if tc.function.name == "submit_answer":
                return AnswerResult(args.get("answer", ""), _enrich(args.get("citations")), True)
            emit({"type": "tool_call", "name": tc.function.name, "args": args})
            result = tools.run_read_tool(tc.function.name, args)
            emit({"type": "tool_result", "name": tc.function.name,
                  "summary": _summarize(tc.function.name, result)})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    return AnswerResult("", [], False)
