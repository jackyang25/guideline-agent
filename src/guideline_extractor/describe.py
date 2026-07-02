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
    max_tokens: int = 64000,
) -> tuple[str, str]:
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=build_messages(image_bytes, raw_text),
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason in ("max_tokens", "refusal"):
        raise RuntimeError(
            f"describe_page: model stopped with stop_reason={message.stop_reason!r} "
            "before producing a usable response"
        )

    text_block = next((b for b in message.content if b.type == "text"), None)
    if text_block is None:
        raise RuntimeError(
            f"describe_page: no text block in response (stop_reason={message.stop_reason!r})"
        )
    return parse_description(text_block.text)
