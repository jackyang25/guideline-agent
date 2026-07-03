import json
import pytest
from page2prose.agent import loop
from page2prose.agent import tools


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


def test_answer_handles_bad_page_number_in_citation(lib):
    client = _FakeClient([
        _response(tool_calls=[_tool_call("1", "submit_answer", {
            "answer": "Screen for TB if cough >= 2 weeks.",
            "citations": [
                {"guideline_id": "APC", "page_number": "n/a"},
                {"guideline_id": "APC", "page_number": None},
                {"guideline_id": "APC", "page_number": 10},
            ]})]),
    ])
    result = loop.answer("what to do for a long cough", client=client)
    assert result.complete is True
    # the valid citation still resolves with its title
    assert {"guideline_id": "APC", "page_number": 10, "title": "Cough"} in result.citations


def _raw_tool_call(cid, name, raw_arguments):
    fn = type("Fn", (), {"name": name, "arguments": raw_arguments})()
    return type("TC", (), {"id": cid, "function": fn})()


def test_answer_handles_malformed_tool_call_arguments(lib):
    client = _FakeClient([
        _response(tool_calls=[_raw_tool_call("1", "search_pages", "{not json")]),
        _response(tool_calls=[_tool_call("2", "submit_answer", {
            "answer": "done", "citations": []})]),
    ])
    result = loop.answer("q", client=client)
    assert result.complete is True
    # the malformed call was fed back as a failed tool result, not a crash
    first_call_msgs = client.chat.completions.calls[-1]["messages"]
    tool_msgs = [m for m in first_call_msgs if m.get("role") == "tool" and m.get("tool_call_id") == "1"]
    assert tool_msgs
    assert "error" in json.loads(tool_msgs[0]["content"])


def test_answer_handles_malformed_submit_answer_arguments(lib):
    client = _FakeClient([
        _response(tool_calls=[_raw_tool_call("1", "submit_answer", "{not json")]),
    ])
    result = loop.answer("q", client=client)
    assert result.complete is False


def test_system_prompt_states_grounding_and_gathering_protocol():
    p = loop.SYSTEM_PROMPT.lower()
    assert "only" in p and "do not use outside" in p          # grounding
    assert "see p.112" in p                                    # follow relevant references
    assert "neighbouring page" in p                            # continuation via neighbors
    assert "do not answer until" in p                          # enough context before answering
    assert "submit_answer" in loop.SYSTEM_PROMPT               # terminal + decline path
    # preserve decision routing (do not flatten branching pages)
    assert "do not flatten" in p
    assert "where it leads" in p
    assert "do not ask the user questions yourself" in p
