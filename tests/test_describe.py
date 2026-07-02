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
    def __init__(self, text, stop_reason="end_turn", content=None):
        self._text = text
        self._stop_reason = stop_reason
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        if self._content is not None:
            content = self._content
        else:
            block = type("Block", (), {"type": "text", "text": self._text})()
            content = [block]
        return type("Msg", (), {"content": content, "stop_reason": self._stop_reason})()


class _FakeMessages:
    def __init__(self, text, stop_reason="end_turn", content=None):
        self._text = text
        self._stop_reason = stop_reason
        self._content = content
        self.captured = None

    def stream(self, **kwargs):
        self.captured = kwargs
        return _FakeStream(self._text, self._stop_reason, self._content)


class _FakeClient:
    def __init__(self, text, stop_reason="end_turn", content=None):
        self.messages = _FakeMessages(text, stop_reason, content)


def test_describe_page_returns_title_and_prose_and_uses_opus():
    client = _FakeClient(json.dumps({"title": "T", "prose": "P"}))
    title, prose = describe_page(client, b"\x89PNG", "raw")
    assert (title, prose) == ("T", "P")
    assert client.messages.captured["model"] == "claude-opus-4-8"
    assert client.messages.captured["thinking"] == {"type": "adaptive"}
    assert client.messages.captured["max_tokens"] == 64000


def test_describe_page_raises_runtime_error_on_max_tokens_stop_reason():
    client = _FakeClient(
        json.dumps({"title": "T", "prose": "P"}), stop_reason="max_tokens"
    )
    with pytest.raises(RuntimeError, match="max_tokens"):
        describe_page(client, b"\x89PNG", "raw")


def test_describe_page_raises_runtime_error_on_refusal_with_empty_content():
    client = _FakeClient("", stop_reason="refusal", content=[])
    with pytest.raises(RuntimeError, match="refusal"):
        describe_page(client, b"\x89PNG", "raw")
