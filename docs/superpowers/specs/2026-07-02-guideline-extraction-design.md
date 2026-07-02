# Guideline Extraction Pipeline — Design

**Date:** 2026-07-02
**Status:** Approved design, pre-implementation
**Scope:** The *extraction* pipeline (PDF → structured page records). The querying agent is
downstream and specced only as far as needed to validate that the extraction output supports it.

---

## 1. Problem & Motivation

Clinical guidelines (e.g. *Adult Primary Care (APC) 2023*, South Africa, 180 pages) encode much of
their meaning in **visual structure** — flowcharts, decision trees, dosing tables, timelines, and
cross-page references ("see p.38"). Forcing that content into a rigid schema loses the logic and
mislabels pages. A prior attempt (`APC_2023_Clinical_tool-EBOOK_chunks.jsonl`, Jan 2026) demonstrated
this failure: it imposed a fixed hierarchy (`section_path`, `chunk_type`) that mislabeled pages and
flattened flowcharts.

**Core idea:** describe each page in rich natural-language **prose** that can *stand in for the page*,
losslessly enough that the page's content and decision logic survive. Downstream, a guideline agent
uses this prose as context — selecting an entry page and bouncing between pages to answer clinical
queries.

## 2. Design Principles

- **Prose over schema.** Structure emerges from the page (via universal Markdown), not from an
  externally imposed taxonomy. This is the explicit correction of the January failure.
- **Lossless substrate + interpreted payload.** Keep the rendered image and verbatim text as
  backstops; the prose is a (re-generatable) interpretation layered on top.
- **Minimal, robust records.** Every field is either lossless substrate, agent-reasoning payload, or
  a stable address. Nothing speculative, no guessed categories.
- **Deterministic where it can be, semantic where it must be.** Physical-page addressing is exact;
  content/named resolution is semantic. Do not blur the two.
- **Multi-guideline from day one.** Build for one guideline; make the structure namespace cleanly to
  many.

## 3. Capture (per page, hybrid)

For each PDF page:

1. **Render** the page to an image (`pNNN.png`) — the lossless visual backstop.
2. **Extract** the raw text layer verbatim — ground truth for doses, drug names, numbers.
3. **Describe**: send **image + raw text** to a vision-capable model (Claude Opus/Sonnet) with the
   fidelity-contract prompt below. The image gives visual/layout understanding; the raw text grounds
   exact values and guards against transcription hallucination.

### 3.1 The Fidelity-Contract Prompt (one general prompt, all pages, all guidelines)

```
Produce ONE self-contained textual description that fully encodes this page.
The page image is NOT available downstream - only your text stands in for it.
Transcribe every heading, label, value, dose, date, and table cell you can read.
Describe the structure and the relationships between elements - reading order,
sequence, decision pathways (if X then Y / go to page N), dependencies, and
groupings - so the page's logic is fully preserved. Use Markdown: tables for
tabular data, headings to mirror the page. Be exhaustive and specific, but do
not infer beyond what is shown; where the page is genuinely unclear, say so
rather than guess.
```

Rationale for each clause:
- *"stands in for it"* — the completeness bar (a clinician could work from the prose alone).
- *"transcribe every … dose … cell"* — verbatim fidelity for clinical safety.
- *"decision pathways (if X then Y / go to page N)"* — makes flowcharts survive as explicit
  conditional logic and seeds cross-page references, **without imposing a node/edge schema**.
- *"Use Markdown: tables … headings"* — light, universal structure. Tables stay grids so a dose never
  drifts to the wrong drug.
- *"do not infer … say so rather than guess"* — the clinical-safety valve; ambiguous flowcharts get
  honest hedging instead of confident invention.

## 4. Data Structure

Two flat levels — a guideline **manifest** (routing) and per-page **records** (payload) — keyed by the
composite `(guideline_id, page_number)`.

### 4.1 Storage layout

```
guidelines/
  APC_2023_ZA/
    manifest.json
    pages/
      p001.json
      p001.png
      p002.json
      p002.png
      ...
  <next_guideline_id>/
    manifest.json
    pages/
      ...
index/                     # derived, built only when scale requires it (§6.3)
  <guideline_id>/...        # BM25 + embedding indexes over prose
```

One folder per guideline = clean namespacing; add a guideline by dropping in a folder.

### 4.2 Guideline manifest (`manifest.json`)

```json
{
  "guideline_id": "APC_2023_ZA",
  "title": "Adult Primary Care (APC) 2023",
  "jurisdiction": "South Africa",
  "publisher": "Western Cape Government Health",
  "version": "2023",
  "effective_date": "2023-01-01",
  "source_file": "APC_2023_Clinical_tool-EBOOK.pdf",
  "page_count": 180,
  "pages": [
    { "page_number": 1,  "title": "Preface" },
    { "page_number": 38, "title": "TB treatment — new patient" }
  ]
}
```

- All manifest fields are **factual**, read off the cover/title page — not a guessed taxonomy.
- `jurisdiction` is a real routing signal (protocols/dosing differ by country); it prevents answering
  a query from the wrong country's book.
- `pages` is a compact **title-map** (page number + title only): loads cheaply so the agent can plan
  navigation and resolve named references without opening every page file.

### 4.3 Page record (`pages/pNNN.json`)

```json
{
  "guideline_id": "APC_2023_ZA",
  "page_number": 38,
  "pdf_index": 40,
  "title": "TB treatment — new patient",
  "prose": "## TB treatment — new patient\n...Markdown; inline refs like 'go to p.42' live here...",
  "image_path": "pages/p040.png",
  "raw_text": "verbatim extracted text layer"
}
```

Field roles:

| Field | Role |
|---|---|
| `guideline_id` + `page_number` | composite key, unique across the whole library |
| `page_number` | the **printed** page number (what references cite) — the chaining/addressing key |
| `pdf_index` | PDF sheet position, when it differs from the printed number |
| `title` | display label **and** named-reference resolution target (not the search key) |
| `prose` | the payload: complete Markdown description; the entry-selection search index |
| `image_path` | lossless visual backstop |
| `raw_text` | verbatim ground truth |

### 4.4 Deliberately excluded

`section_path`, `chunk_type`, `category`, `cross_refs`, `keywords`, and `embedding` are **not** in the
source records — each is either a guessed taxonomy (brittle), a fact extracted twice (drift risk), or a
derived artifact. Search indexes are derived from `prose` into `index/` only when scale requires it.

### 4.5 Page numbering (printed vs PDF)

References cite **printed** page numbers, which may differ from PDF sheet index (cover, front matter,
offsets). Resolve this **at extraction**, not at query time:
- `page_number` = printed number (read from the page corner).
- `pdf_index` = sheet position, stored when it differs.
- **QC check:** printed numbers should increase monotonically; flag any page that breaks the sequence
  for human review against its image.
- Unnumbered pages (tab dividers, roman-numeral front matter) fall back to `pdf_index`; references
  rarely target these.

## 5. Extraction Output — Definition of Done

- One `manifest.json` per guideline with all factual fields + complete `pages` title-map.
- One `pNNN.json` + `pNNN.png` per physical page (the physical page is the addressable unit; **no**
  semantic merging/splitting, which would break page-number addressing).
- `prose` passes the fidelity bar: all headings/values/doses transcribed, tables as Markdown tables,
  decision logic as explicit `if X → then Y / go to p.N`, ambiguities hedged not invented.
- Page-number QC passed or flagged.

## 6. Querying Model (downstream — requirements the extraction must support)

Specced here only to confirm the extraction output is sufficient. Full agent design is a separate spec.

### 6.1 Two-level retrieval

1. **Route (across guidelines):** scan the small set of `manifest.json` files; match query against
   `title` + `jurisdiction` to pick the relevant guideline(s). Scales to hundreds of guidelines.
2. **Select + chain (within a guideline):** per §6.2.

### 6.2 Fan-out, never one giant call

- **Never** put multiple guidelines in one call (cost + cross-jurisdiction correctness risk +
  long-context recall loss).
- **Fan out** one call per routed guideline, in parallel; a small reduce step merges/compares answers.
- **Within each guideline call, pick by size:**
  - Fits the context window (a single book like APC): **full-context mode** — load all page prose;
    entry-selection and named-reference resolution are trivial because everything is present.
  - Too large / many guidelines: **chaining mode** — retrieval + navigation per §6.3–6.4.
- For the current single-guideline state, all tiers collapse to one full-context call. The pipeline
  generalizes into fan-out without rework.

### 6.3 Entry-page selection (chaining mode)

- **Index over `prose`, never over `title`** (query terms may live only in a page's body).
- **Hybrid** retrieval (BM25 + embeddings) over prose → top-k **candidates** (never top-1), each with
  `title` + snippet. Lexical catches exact clinical terms (drug/test names); semantic catches
  paraphrase.
- The agent **reads the full prose** of top candidates and **confirms against the specific need**
  (e.g. "TB *treatment regimen*", not merely "about TB") before committing — so a coarse first pass
  never silently starts on the wrong page, and topically-relevant-but-wrong pages are rejected.
- Indexes are derived from `prose` into `index/`; source records unchanged.

### 6.4 Bouncing between pages — agent tools

| Tool | Handles | Mechanism | Reliability |
|---|---|---|---|
| `get_page(guideline_id, page_number)` | exact page refs ("see p.38"); following current page | deterministic lookup on printed `page_number`; returns prose + neighbor breadcrumbs `{prev,next: {page_number,title}}` | exact |
| `search_prose(guideline_id, query)` | named/content refs ("see Diabetes section"); multi-topic pages | hybrid over prose → ranked candidates + snippets | high (content-based) |

Resolution strategy:
- **Reference resolution is the agent reasoning over the title-map + tools — one resolver, all forms**
  (page number, section name, tab, box/table label). No per-form parsing, no extracted link table.
- **Deterministic-first, semantic-fallback:** try `get_page` for a cited number; if the returned
  page's content clearly doesn't match the reference's intent (page-number captured wrong), fall back
  to `search_prose`.
- **Candidates + intent disambiguation:** when multiple pages match a topic (TB spans screening /
  treatment / DR-TB), the agent disambiguates using the reference's *intent phrase* and candidate
  titles, then verifies against the need.
- **Neighbor expansion:** content spanning consecutive pages is read via `get_page(n±1)` (no separate
  tool — `page_number` is contiguous) when a page looks like part of a spread or the answer clearly
  continues. Distinct from `search_prose` (topical) — this is physical adjacency.

## 7. Build Order

1. **Extraction pipeline** (this spec) — capture + prose + records + manifest + QC. Validate prose
   quality on the APC guideline before anything else.
2. Querying agent (separate spec) — routing, fan-out/reduce, tools, full-context → hybrid transition.

## 8. Open Items

- None blocking. The querying agent (§6) is the next design once extraction output is validated.
