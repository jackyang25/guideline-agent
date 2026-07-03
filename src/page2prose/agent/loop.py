import json
import os
from dataclasses import dataclass, field

from .. import library
from . import tools

SYSTEM_PROMPT = (
    "You answer clinical questions using ONLY the provided guideline tools. Do not use outside "
    "medical knowledge and do not infer beyond what the pages say.\n"
    "Quote dosages, drug names, and numeric values EXACTLY as printed on the page - copy them "
    "verbatim; never paraphrase, round, reformat, or re-unit a number (e.g. do not turn "
    "'2.5mg IM if > 65 years' into '2.5 mrs').\n\n"
    "Finding the answer:\n"
    "- Call list_guidelines and pick the relevant guideline(s) by jurisdiction and title.\n"
    "- Use search_pages to find candidate pages, then get_page to read the most relevant one.\n\n"
    "Gather enough context before you answer:\n"
    "- If a page only partly answers the question, open more of the search candidates.\n"
    "- If a page you read points to another page (e.g. 'see p.112') and that reference is relevant to "
    "the question, open it with get_page. Do not chase references that are not relevant.\n"
    "- If a flowchart, table, or the answer runs to the edge of a page or is marked as continued, open "
    "the neighbouring page (get_page on the prev/next page number).\n"
    "- Do not answer until the pages you have actually read fully cover the question.\n\n"
    "Preserve decision routing:\n"
    "- Many pages are decision/triage pages that branch conditionally (e.g. 'if stress or anxiety, "
    "see p.86'; 'if pulse >= 100, check TSH'; 'refer to doctor if abnormal'). When the relevant page "
    "branches, present each branch AND where it leads - the destination page, or the instruction such "
    "as 'refer to doctor' or 'check TSH' - as distinct options in your answer.\n"
    "- Do not flatten a branching page into one undifferentiated summary, and do not assume which "
    "branch applies to the patient. Surface the options with their destinations so the caller can "
    "choose the relevant one and ask a follow-up. You do not ask the user questions yourself.\n\n"
    "Finishing:\n"
    "- Cite every page you used. Call submit_answer with the answer and its citations.\n"
    "- If the guidelines do not address the question, no guideline is relevant, or you cannot find "
    "enough to answer confidently, call submit_answer saying so plainly - do not guess."
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
