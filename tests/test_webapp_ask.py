import json
from fastapi.testclient import TestClient
from page2prose import webapp
from page2prose.agent.loop import AnswerResult


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


def test_index_has_ask_and_browse_tabs(monkeypatch, tmp_path):
    monkeypatch.setenv("GE_OUTPUT_ROOT", str(tmp_path))
    html = TestClient(webapp.app).get("/").text
    assert 'id="tab-ask"' in html
    assert 'id="tab-browse"' in html
    assert 'id="view-ask"' in html
    assert 'id="view-browse"' in html
