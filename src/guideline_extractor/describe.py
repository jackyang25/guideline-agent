import base64
import json
import os

DEFAULT_MODEL = "gpt-5.5"

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
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
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
    model: str | None = None,
    max_completion_tokens: int = 32000,
) -> tuple[str, str]:
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_completion_tokens,
        messages=build_messages(image_bytes, raw_text),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "page_description",
                "schema": _SCHEMA,
                "strict": True,
            },
        },
    )
    choice = response.choices[0]
    if choice.finish_reason in ("length", "content_filter"):
        raise RuntimeError(
            f"describe_page: model stopped with finish_reason={choice.finish_reason!r} "
            "before producing a usable response"
        )
    content = choice.message.content
    if content is None:
        raise RuntimeError(
            f"describe_page: no content in response (finish_reason={choice.finish_reason!r})"
        )
    return parse_description(content)
