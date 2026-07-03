import base64
import json

import pytest
from page2prose.describe import (
    build_messages,
    parse_description,
    describe_page,
    FIDELITY_PROMPT,
)


def test_build_messages_has_text_and_data_uri_image():
    msgs = build_messages(b"\x89PNGdata", "raw ground truth")
    content = msgs[0]["content"]
    assert content[0]["type"] == "text"
    assert "raw ground truth" in content[0]["text"]
    assert FIDELITY_PROMPT in content[0]["text"]
    assert content[1]["type"] == "image_url"
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.standard_b64encode(b"\x89PNGdata").decode() in url


def test_parse_description_extracts_title_and_prose():
    text = json.dumps({"title": "Cough", "prose": "## Cough\n..."})
    assert parse_description(text) == ("Cough", "## Cough\n...")


def test_parse_description_raises_on_missing_keys():
    with pytest.raises(ValueError):
        parse_description(json.dumps({"title": "x"}))


class _FakeCompletions:
    def __init__(self, content, finish_reason="stop"):
        self._content = content
        self._finish_reason = finish_reason
        self.captured = None

    def create(self, **kwargs):
        self.captured = kwargs
        message = type("Msg", (), {"content": self._content})()
        choice = type("Choice", (), {"message": message, "finish_reason": self._finish_reason})()
        return type("Resp", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, content, finish_reason="stop"):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content, finish_reason)})()


def test_describe_page_returns_title_and_prose_and_uses_json_schema(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = _FakeClient(json.dumps({"title": "T", "prose": "P"}))
    title, prose = describe_page(client, b"\x89PNG", "raw")
    assert (title, prose) == ("T", "P")
    kw = client.chat.completions.captured
    assert kw["model"] == "gpt-5.5"  # default model
    assert kw["response_format"]["type"] == "json_schema"
    assert kw["response_format"]["json_schema"]["strict"] is True


def test_describe_page_honors_openai_model_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    client = _FakeClient(json.dumps({"title": "T", "prose": "P"}))
    describe_page(client, b"\x89PNG", "raw")
    assert client.chat.completions.captured["model"] == "gpt-4o"


def test_describe_page_explicit_model_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    client = _FakeClient(json.dumps({"title": "T", "prose": "P"}))
    describe_page(client, b"\x89PNG", "raw", model="o4-mini")
    assert client.chat.completions.captured["model"] == "o4-mini"


def test_describe_page_raises_on_length_truncation():
    client = _FakeClient(None, finish_reason="length")
    with pytest.raises(RuntimeError):
        describe_page(client, b"\x89PNG", "raw")


def test_describe_page_raises_on_content_filter():
    client = _FakeClient(None, finish_reason="content_filter")
    with pytest.raises(RuntimeError):
        describe_page(client, b"\x89PNG", "raw")


def test_fidelity_prompt_flags_continuation_and_cross_refs():
    p = FIDELITY_PROMPT.lower()
    assert "continues on the next page" in p or "cut off" in p
    assert "see p.112" in p  # record cross-references as printed


def test_detect_metadata_parses_fields(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    import json as _json
    from page2prose import describe
    payload = _json.dumps({"title": "Adult Primary Care 2023", "jurisdiction": "South Africa",
                           "publisher": "WCGH", "version": "2023", "effective_date": None})
    client = _FakeClient(payload)
    meta = describe.detect_metadata(client, b"\x89PNG", "raw")
    assert meta["title"] == "Adult Primary Care 2023"
    assert meta["jurisdiction"] == "South Africa"
    assert meta["effective_date"] is None
    kw = client.chat.completions.captured
    assert kw["response_format"]["json_schema"]["name"] == "guideline_metadata"


def test_detect_metadata_returns_nulls_on_truncation():
    from page2prose import describe
    client = _FakeClient(None, finish_reason="length")
    meta = describe.detect_metadata(client, b"\x89PNG", "raw")
    assert meta == {"title": None, "jurisdiction": None, "publisher": None,
                    "version": None, "effective_date": None}
