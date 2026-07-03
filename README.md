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

Open http://127.0.0.1:8000. On the **Ask** tab, ask a question and watch the agent navigate to a cited
answer. On the **Browse** tab, upload a PDF and hit **Extract** — leave **ID**/**Title** blank to have
them detected from the cover page (or set them to override); optionally set **Limit** (first N pages,
for a cheap smoke test) and **Workers** (parallel pages, default 25). Progress shows live. Then pick a
guideline in **View** and click a page to inspect its data block — Prose (rendered), Raw text, **JSON**
(the exact stored record), and the Page image.

Stop with `Ctrl+C`. If a start fails with "address already in use", a previous server is still running:
`lsof -ti:8000 | xargs kill`.

## Run the CLI

```bash
.venv/bin/python -m page2prose.cli PDF OUT_ROOT --limit 3 --concurrency 25
# id/title/jurisdiction/version are detected from the cover page by default;
# pass --guideline-id / --guideline-title / --jurisdiction / --version to override.
# Output is written to OUT_ROOT/<id>.
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
