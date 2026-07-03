from page2prose import ask
from page2prose.agent.loop import AnswerResult


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
