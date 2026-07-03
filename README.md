# Guideline Extractor

Turn a clinical guideline PDF into one prose-per-page record set that stands in for the pages —
faithful Markdown descriptions (flowcharts, tables, decision paths) plus the source image and raw
text — so a downstream agent can index and navigate it.

## Setup (one time)

```bash
cd guideline-agent
python3 -m venv .venv
.venv/bin/pip install -e ".[ui]"     # use ".[dev]" if you also want to run tests
```

Create `.env` (copy from `.env.example`) and add your key:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.5
```

## Run the web UI

```bash
.venv/bin/python -m page2prose.webapp
```

Open http://127.0.0.1:8000. Upload a PDF, set **ID** / **Title**, optionally **Limit** (first N pages,
for a cheap smoke test) and **Workers** (parallel pages, default 25), then **Extract**. Progress shows
live. Pick a guideline in **View** and click a page to inspect its data block — Prose (rendered), Raw
text, **JSON** (the exact stored record), and the Page image.

Stop with `Ctrl+C`. If a start fails with "address already in use", a previous server is still running:
`lsof -ti:8000 | xargs kill`.

## Run the CLI

```bash
.venv/bin/python -m page2prose.cli PDF OUT_DIR \
  --guideline-id APC_2023_ZA --guideline-title "Adult Primary Care (APC) 2023" \
  --jurisdiction "South Africa" --version 2023 --limit 3 --concurrency 25
```

## Output

```
guidelines/<id>/
  manifest.json              # guideline metadata + page map (page_number/title/pdf_index)
  pages/pNNN.json            # per-page record: page_number, pdf_index, title, prose, raw_text, image_path
  pages/pNNN.png             # rendered source page
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Design

See `docs/superpowers/specs/2026-07-02-guideline-extraction-design.md` for the full spec (capture,
data structure, page-number calibration/QC, and the Phase 2 querying-agent design).
