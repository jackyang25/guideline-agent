# Guideline Answering Agent — Design (Phase 2)

**Status:** Approved design, pre-implementation.
**Depends on:** Phase 1 extraction output (`guidelines/<id>/manifest.json` + `pages/pNNN.json`).
**Scope:** A grounded answering agent that takes a clinical query, navigates the extracted
guidelines with tools, and returns an answer cited to specific pages — or declines when the
guidelines don't cover it. Reads the `guidelines/` folder; does not modify extraction.

---

## 1. Problem & approach

Phase 1 produced rich per-page **prose** (flowcharts linearized, decision paths spelled out) plus a
manifest page-map per guideline. Phase 2 turns that into answers.

It is **one grounded answering agent**, not a separate retrieval system: retrieval and navigation are
*tools* the agent calls. The agent takes a query, calls tools over `guidelines/`, reads prose, decides
when it has enough, and submits a final answer with page citations — closing the loop itself. A
retrieval-only system would hand back pages and make the caller reason; that leaves the prose design's
value on the table.

**Grounding is the core constraint:** the agent answers *only* from text the tools return, cites the
pages it used, and says "the guidelines don't cover this" rather than fall back on the model's own
medical knowledge. It reasons and synthesizes, but only over what the guidelines actually say.

## 2. Retrieval approach (embeddings-free)

Tool-based navigation over the on-disk output — no vector store, no embedding calls, no new service:

- The agent sees the available guidelines (`list_guidelines`) and routes by title/jurisdiction.
- It finds candidate pages with **BM25 over `prose`** (`search_pages`) — catches body content that
  titles miss; in-memory (`rank_bm25`), built lazily per guideline and cached.
- It reads full prose with `get_page`, which also returns neighbor breadcrumbs so it can expand to
  adjacent pages when content continues.

This is the spec-sanctioned "hybrid, embeddings deferred until scale" position: BM25 now; the same tool
signatures are the seam to add embeddings later for a large multi-guideline library, without changing
the agent.

**Neighbor expansion** is agent-driven: `get_page` returns `prev`/`next` (number + title), computed at
query time from the manifest map. The agent opens `n±1` when a flowchart/table/answer clearly
continues. No automatic prefetching. Distinct from `search_pages` (topical) — this is physical
adjacency.

## 3. Tools

All read-only over `guidelines/` (rooted at `GE_OUTPUT_ROOT`, default `./guidelines`).

| Tool | Signature | Returns |
|---|---|---|
| `list_guidelines` | `()` | `[{guideline_id, title, jurisdiction, page_count}]` |
| `search_pages` | `(guideline_id, query, k=8)` | ranked `[{page_number, title, snippet}]` (BM25 over prose) |
| `get_page` | `(guideline_id, page_number)` | `{page_number, title, prose, prev, next}` (`prev`/`next` = `{page_number, title}` or null) |
| `submit_answer` | `(answer, citations)` | **terminal** — ends the loop; `citations = [{guideline_id, page_number}]` |

- `search_pages` ranks with BM25; a `snippet` is a short prose excerpt around the match so the agent
  can triage without opening each page.
- `submit_answer` is how the agent closes the loop: a structured final answer + the pages it used.
  This guarantees citations rather than parsing free text.
- Tool errors (unknown guideline/page, empty library) are returned to the model as tool results, not
  raised, so it can recover or decline.

## 4. The loop

`answer(query, model=None, max_turns=12, on_event=None) -> AnswerResult`:
1. Build messages: system prompt (§5) + user query. Offer the four tools.
2. Call the model (OpenAI Chat Completions, tool-calling). Model default from `OPENAI_MODEL`
   (`gpt-5.5`).
3. Execute each tool call the model makes; append results; repeat.
4. When the model calls `submit_answer`, return the `AnswerResult`.
5. If `max_turns` is hit without `submit_answer`, return a result flagged incomplete (the agent ran
   out of turns) — never fabricate an answer.

`AnswerResult`: `{answer: str, citations: [{guideline_id, page_number, title}], complete: bool}`
(titles resolved from the manifest for display/linking).

**Traceability (live navigation).** So a caller can watch the agent work, the loop takes an optional
`on_event(event: dict)` callback, invoked as it navigates:
- `{"type": "tool_call", "name", "args"}` — before executing a tool the model requested.
- `{"type": "tool_result", "name", "summary"}` — after, `summary` a short human string (e.g.
  `"3 pages: p.34, p.38"` or `"p.34 — Cough"`).
The callback is the trace source for both entrypoints (§6): the CLI prints the steps, the UI streams
them live. `submit_answer` is not emitted as a tool event — it becomes the returned `AnswerResult` /
the stream's terminal `done` event.

## 5. System prompt (grounding contract)

One prompt, provider-agnostic in spirit:
- You answer clinical questions **only** from the provided guideline tools. Do not use outside medical
  knowledge or infer beyond what the pages say.
- Use `list_guidelines` to see what's available and pick the relevant one(s) by jurisdiction/title.
- Use `search_pages` to find an entry page, `get_page` to read it, and open neighbor pages when the
  content or answer continues.
- Cite every page you used. Call `submit_answer` with the answer and citations to finish.
- If the guidelines don't address the query (or no guideline is relevant), `submit_answer` saying so
  plainly — do not guess.

## 6. Entrypoints

- **CLI:** `python -m guideline_extractor.ask "QUERY"` → prints each navigation step live (from
  `on_event`), then the answer and a Sources list (`guideline_id p.N — title`). The testable core.
- **UI:** a query box + `POST /api/ask` in the existing web app that **streams NDJSON** (same pattern
  as extraction progress): `tool_call` / `tool_result` events as the agent navigates, then a terminal
  `{"type": "done", "answer", "citations", "complete"}` (or `{"type": "error", "detail"}`). The UI
  shows the live trace ("search_pages: cough → 3 pages", "get_page: p.34 — Cough"), then renders the
  answer with its cited pages listed.

Both load `.env` (needs `OPENAI_API_KEY`).

## 7. Code shape

| Module | Responsibility |
|---|---|
| `library.py` | read-only access to `guidelines/`: `list_guidelines()`, `load_manifest(id)`, `load_page(id, page_number)`, neighbor lookup. Pure, no OpenAI. |
| `agent/tools.py` | the four tool implementations + BM25 index (lazy, cached per guideline), built on `library.py`. |
| `agent/loop.py` | `answer(...)`, the tool-calling loop; the system prompt; `AnswerResult`. |
| `ask.py` | CLI entrypoint. |
| `webapp.py` | add `POST /api/ask` + a query box in `INDEX_HTML`. |

`library.py` is a new shared reader; the webapp currently reads the same data inline — left as-is to
stay focused, but this is where the two would converge later.

## 8. Error handling
- No `OPENAI_API_KEY` → clear error at call time (surfaced in CLI/UI).
- Empty `guidelines/` → `list_guidelines` returns `[]`; the agent declines ("no guidelines available").
- Unknown guideline/page in a tool call → error returned as a tool result; the agent recovers.
- `max_turns` exhausted → `complete=false`, no fabricated answer.

## 9. Testing
- **`library.py`**: list/load/neighbor lookup against a fixture `guidelines/` dir.
- **Tools**: BM25 ranking returns the expected page for a query; `get_page` neighbors correct at edges;
  unknown ids return errors not exceptions.
- **Loop**: a fake OpenAI client scripted to (a) call `search_pages` → `get_page` → `submit_answer`
  and (b) decline via `submit_answer` — verifies tools execute, citations flow through, and both the
  answer and decline paths work. No network.
- **Entrypoints**: CLI prints answer+sources (loop stubbed); `/api/ask` returns the JSON (loop stubbed).

## 10. Out of scope (deferred)
Embeddings / vector store (BM25 is enough for now); cross-guideline fan-out + reduce (the agent handles
one-or-few guidelines via `list_guidelines` routing; parallel fan-out is a later optimization);
conversational multi-turn memory (single query in, single answer out); answer-quality evals.
