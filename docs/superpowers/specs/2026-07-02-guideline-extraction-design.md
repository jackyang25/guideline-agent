# Guideline Extraction Pipeline — Spec

**Status:** Extraction pipeline implemented and shipped. Querying agent: design only (Phase 2).
**Scope:** Phase 1 turns a guideline PDF into one prose-per-page record set that stands in for the
pages. Phase 2 (the querying agent) is designed here only far enough to confirm the extraction
output supports it.

---

## 1. Problem

Clinical guidelines (e.g. *Adult Primary Care (APC) 2023*, South Africa, 180 pages) encode much of
their meaning in **visual structure** — flowcharts, decision trees, dosing tables, timelines, and
cross-page references ("see p.38"). Forcing that into a rigid schema loses the logic and mislabels
pages. Instead we describe each page in rich natural-language **prose** that can *stand in for the
page*, and keep the source image + raw text as lossless backstops. Downstream, an agent uses the
prose as context — selecting an entry page and moving between pages to answer clinical questions.

## 2. Design principles

- **Prose over schema.** Structure emerges from the page via universal Markdown, not an imposed
  taxonomy. This is the correction of an earlier attempt that forced a fixed hierarchy and mislabeled
  pages.
- **Lossless substrate + interpreted payload.** Keep the rendered page image and verbatim text as
  backstops; the prose is a re-generatable interpretation on top.
- **Minimal, robust records.** Every field is lossless substrate, agent-reasoning payload, or a stable
  address. No guessed categories.
- **Deterministic where possible, semantic where necessary.** Physical-page addressing is exact;
  content/named resolution is semantic.
- **Multi-guideline from day one.** One folder per guideline; the structure namespaces cleanly.

## 3. Capture (per page, hybrid)

For each PDF page (`render.py`):
1. Render the page to a PNG (PyMuPDF) — the lossless visual backstop.
2. Extract the raw text layer verbatim — ground truth for doses/drug names.
3. Send **image + raw text** to a vision model with the fidelity-contract prompt (`describe.py`).

**Provider:** the OpenAI Chat Completions API with a vision (`image_url` data-URI) message and **strict
JSON-schema structured output** returning `{title, prose}`. Model is configurable via the
`OPENAI_MODEL` env var (default `gpt-5.5`). `finish_reason` of `length` or `content_filter` (or missing
content) raises a clear error rather than returning a truncated/empty record. `describe.py` is the
source of truth for the API surface. (The original design targeted Claude; the pipeline is otherwise
provider-agnostic — only `describe.py` and the client construction in `pipeline.py` are provider-specific.)

**Fidelity-contract prompt** (one general prompt for every page, every guideline):
```
Produce ONE self-contained textual description that fully encodes this page. The page image is NOT
available downstream - only your text stands in for it. Transcribe every heading, label, value, dose,
date, and table cell you can read. Describe the structure and the relationships between elements -
reading order, sequence, decision pathways (if X then Y / go to page N), dependencies, and groupings -
so the page's logic is fully preserved. Use Markdown: tables for tabular data, headings to mirror the
page. Be exhaustive and specific, but do not infer beyond what is shown; where the page is genuinely
unclear, say so rather than guess.
Return a JSON object with two fields: "title" (a short factual label of what this page covers) and
"prose" (the full Markdown description).
```

**Parallelism:** pages are independent and described concurrently (`ThreadPoolExecutor`, default **25**
workers — one page per worker). Results are reassembled in page order. Effective concurrency is bounded
by the account's token-per-minute limit; the SDK retries 429s with backoff. Configure via the UI
**Workers** field or the CLI `--concurrency` flag. Progress is reported per completed page (`on_page`
callback) and streamed to the UI live.

## 4. Data structure

Two flat levels, keyed by the composite `(guideline_id, page_number)`.

### Storage layout
```
guidelines/
  APC_2023_ZA/
    manifest.json
    pages/
      p001.json
      p001.png
      ...
```
One folder per guideline. On-disk filenames use zero-padded **`pdf_index`** (`p001`) — always unique and
independent of page-number calibration, so the filesystem never collides. The manifest map resolves a
printed `page_number` → sheet.

### Manifest (`manifest.json`)
```json
{
  "guideline_id": "APC_2023_ZA",
  "title": "Adult Primary Care (APC) 2023",
  "jurisdiction": "South Africa",
  "publisher": null,
  "version": "2023",
  "effective_date": null,
  "source_file": "APC_2023_Clinical_tool-EBOOK.pdf",
  "page_count": 180,
  "pages": [ { "page_number": 34, "title": "Cough — approach to the adult", "pdf_index": 34 } ]
}
```
`jurisdiction` is a real routing signal (protocols/dosing differ by country). `pages` is a compact
title-map for cheap navigation/planning.

### Page record (`pages/pNNN.json`)
```json
{
  "guideline_id": "APC_2023_ZA",
  "page_number": 34,
  "pdf_index": 34,
  "title": "Cough — approach to the adult",
  "prose": "Markdown; inline references like 'go to p.38' live here",
  "image_path": "pages/p034.png",
  "raw_text": "verbatim text layer"
}
```

**Deliberately excluded** (each was guessed taxonomy, extracted-twice, or a derived artifact):
`section_path`, `chunk_type`, `category`, `cross_refs`, `keywords`, `embedding`. Search indexes, if
ever needed, are derived from `prose` downstream — the source records stay clean.

### Page numbering (printed vs PDF)
References cite **printed** page numbers, which may differ from the PDF sheet index. Resolved once, at
extraction, in `pagemap.resolve_page_numbers` (no per-query work):
1. `pdf_index` = the deterministic 1-based sheet counter.
2. Calibrate a single printed↔sheet offset from the detected printed numbers.
3. `page_number = pdf_index + offset`.
4. **QC gate:** flag any page where printed numbering is not strictly increasing, or where a detected
   printed number disagrees with its calibrated page number (restart, mis-detected offset). Flags are
   surfaced in the UI/CLI for human review against the page image.

At query time the agent uses one key, `page_number`; `pdf_index` is internal (locates the image, QC
fallback).

### Resilience & resume
Records are written per page as they complete. A page whose model call fails (truncation, refusal, …)
is recorded in the returned `failed` list and left unwritten — it does **not** abort the batch; the
manifest is written for the pages that succeeded. Re-running the same guideline **resumes**: pages that
already have a record are reused (no model call), so only missing/failed pages are retried.
`extract(...)` returns `(manifest, flags, failed)`. Auto-derived ids never overwrite a different
guideline — if the target folder exists, the id gets a numeric suffix; an explicit id writes into the
same folder (intentional re-extract/resume).

**Known gap:** roman-numeral / unnumbered front-matter pages (detected number `None`) receive a
calibrated arabic `page_number` without a QC flag — reliably distinguishing them from body pages whose
number wasn't on its own line needs roman-numeral detection (not yet built).

## 5. Querying model (Phase 2 — design only, not built)

Documented to confirm the extraction output supports it. Not implemented.

- **Route (across guidelines):** scan the manifests; match query against `title` + `jurisdiction`.
- **Fan out, never one giant call:** one call per routed guideline, in parallel, then a small reduce.
  Never combine guidelines in a single call (cost + cross-jurisdiction correctness + long-context recall).
- **Within a guideline:** full-context when it fits the window; otherwise **hybrid retrieval over
  `prose`** (BM25 + embeddings) → top-k candidates → the agent reads their full prose and confirms
  against the specific need before committing.
- **Bouncing between pages — two agent tools:**
  - `get_page(guideline_id, page_number)` — deterministic lookup for cited page refs; returns prose plus
    neighbor breadcrumbs.
  - `search_prose(guideline_id, query)` — hybrid search over prose for named/content refs and
    multi-topic pages.
  Reference resolution is the agent reasoning over the title-map + tools (one resolver for all forms —
  page number, section name, tab, box/table label). Deterministic-first, semantic-fallback; expand to
  adjacent pages via `get_page(n±1)` when content continues.

## 6. Running it

**Web UI** (upload → extract with live progress → inspect each page's data block):
```
.venv/bin/python -m page2prose.webapp   # http://127.0.0.1:8000
```
Viewer tabs per page: Prose (rendered), Raw text, JSON (the exact stored record), Page image.

**CLI:**
```
.venv/bin/python -m page2prose.cli PDF OUT_ROOT --limit 3 --concurrency 25
```
Guideline **id, title, jurisdiction, and version are auto-detected from the front matter** (a single
`detect_metadata` call: the cover image + the text of the first ~3 pages, where bibliographic metadata
lives; absent fields stay `null`, dates normalized to `YYYY-MM-DD`/`YYYY`); `id` defaults to a slug of
the title. Pass `--guideline-id` / `--guideline-title` /
`--jurisdiction` / `--version` to override any of them. Output is written to `OUT_ROOT/<id>`.

## 7. Configuration (`.env`)
```
OPENAI_API_KEY=sk-...        # required for extraction (viewing works without it)
OPENAI_MODEL=gpt-5.5         # any OpenAI vision + structured-output model
# GE_OUTPUT_ROOT=./guidelines  # where extractions are written (default ./guidelines)
```
Workers are set per-run (UI Workers field / CLI `--concurrency`, default 25). `--limit` / UI Limit
process only the first N pages for a cheap smoke test.

## 8. Modules
| Module | Responsibility |
|---|---|
| `render.py` | PDF → per-page PNG + raw text |
| `pagemap.py` | detect/calibrate printed numbers, QC gate (`resolve_page_numbers`) |
| `describe.py` | fidelity-contract prompt → `{title, prose}` via OpenAI |
| `storage.py` | manifest + `pages/pNNN.{json,png}` layout |
| `pipeline.py` | orchestration: render → resolve → describe (parallel) → write |
| `cli.py` / `webapp.py` | entrypoints (both load `.env`) |
