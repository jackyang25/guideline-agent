import base64
import json
import os
import re

DEFAULT_MODEL = "gpt-5.5"

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR = re.compile(r"(?:19|20)\d{2}")


def _clean_str(value) -> str | None:
    """Strip whitespace; empty / 'null' / 'n/a' become None so fields are symmetric."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("null", "none", "n/a", "na", "unknown"):
        return None
    return s


def _normalize_date(value) -> str | None:
    """Coerce a date to 'YYYY-MM-DD' or 'YYYY'; else pull out a 4-digit year; else None."""
    s = _clean_str(value)
    if s is None:
        return None
    if _ISO_DATE.match(s):
        return s
    year = _YEAR.search(s)
    return year.group(0) if year else None

FIDELITY_PROMPT = (
    "Produce ONE self-contained textual description that fully encodes this "
    "page. The page image is NOT available downstream - only your text stands "
    "in for it. Transcribe every heading, label, value, dose, date, and table "
    "cell you can read. Describe the structure and the relationships between "
    "elements - reading order, sequence, decision pathways (if X then Y / go "
    "to page N), dependencies, and groupings - so the page's logic is fully "
    "preserved. Use Markdown: tables for tabular data, headings to mirror the "
    "page. Be exhaustive and specific, but do not infer beyond what is shown; "
    "where the page is genuinely unclear, say so rather than guess. "
    "If a flowchart, table, or section appears cut off at the page edge, or "
    "continued from or onto another page, say so explicitly (e.g. 'this "
    "flowchart continues on the next page'). Record any cross-reference to "
    "another page exactly as printed (e.g. 'see p.112').\n\n"
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


METADATA_PROMPT = (
    "This is the cover / title page of a clinical guideline document. From what is "
    "printed on it, extract these fields in a CONSISTENT, NORMALISED form. Use ONLY "
    "what is actually shown; set a field to null if it is not printed. Do not guess.\n"
    "- title: the document's title, as printed.\n"
    "- jurisdiction: the country or region it applies to, as a plain country/region "
    "name (e.g. 'South Africa'), not an abbreviation.\n"
    "- publisher: the issuing organisation's name, as printed.\n"
    "- version: the edition or version, as printed (e.g. '2023', '2nd edition').\n"
    "- effective_date: the publication/effective date as ISO 8601 'YYYY-MM-DD'; if "
    "only a year (or month and year) is shown, give just 'YYYY'; never write month "
    "names or free text; null if no date is shown.\n\n"
    "Return a JSON object with fields \"title\", \"jurisdiction\", \"publisher\", "
    "\"version\", \"effective_date\" (each a string or null)."
)

_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"]},
        "jurisdiction": {"type": ["string", "null"]},
        "publisher": {"type": ["string", "null"]},
        "version": {"type": ["string", "null"]},
        "effective_date": {"type": ["string", "null"]},
    },
    "required": ["title", "jurisdiction", "publisher", "version", "effective_date"],
    "additionalProperties": False,
}


def build_metadata_messages(image_bytes: bytes, raw_text: str) -> list[dict]:
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    text = (
        METADATA_PROMPT
        + "\n\nExact text extracted from this page (ground truth):\n"
        + raw_text
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }
    ]


def detect_metadata(
    client,
    image_bytes: bytes,
    raw_text: str,
    model: str | None = None,
    max_completion_tokens: int = 4000,
) -> dict:
    """Read the cover page and return {title, jurisdiction, publisher, version,
    effective_date} (each str or None). Never raises for a normal response."""
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_completion_tokens,
        messages=build_metadata_messages(image_bytes, raw_text),
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "guideline_metadata", "schema": _METADATA_SCHEMA, "strict": True},
        },
    )
    choice = response.choices[0]
    if choice.finish_reason in ("length", "content_filter") or choice.message.content is None:
        return {k: None for k in _METADATA_SCHEMA["properties"]}
    data = json.loads(choice.message.content)
    return {
        "title": _clean_str(data.get("title")),
        "jurisdiction": _clean_str(data.get("jurisdiction")),
        "publisher": _clean_str(data.get("publisher")),
        "version": _clean_str(data.get("version")),
        "effective_date": _normalize_date(data.get("effective_date")),
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
