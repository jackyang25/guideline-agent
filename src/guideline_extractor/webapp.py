import json
import os
import queue
import tempfile
import threading
from pathlib import Path

import markdown as md
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from .pipeline import extract

load_dotenv()  # read OPENAI_API_KEY (and GE_OUTPUT_ROOT) from a .env file if present

app = FastAPI(title="Guideline Extractor")


def output_root() -> Path:
    return Path(os.environ.get("GE_OUTPUT_ROOT", "guidelines")).resolve()


def _guideline_dir(guideline_id: str) -> Path:
    root = output_root()
    d = (root / guideline_id).resolve()
    if root not in d.parents and d != root:
        raise HTTPException(status_code=400, detail="invalid guideline id")
    if not (d / "manifest.json").exists():
        raise HTTPException(status_code=404, detail="guideline not found")
    return d


@app.get("/api/guidelines")
def list_guidelines() -> list[dict]:
    root = output_root()
    if not root.exists():
        return []
    out = []
    for child in sorted(root.iterdir()):
        manifest = child / "manifest.json"
        if manifest.is_file():
            m = json.loads(manifest.read_text())
            out.append(
                {
                    "guideline_id": m["guideline_id"],
                    "title": m["title"],
                    "page_count": m["page_count"],
                }
            )
    return out


@app.get("/api/guidelines/{guideline_id}/manifest")
def get_manifest(guideline_id: str) -> JSONResponse:
    d = _guideline_dir(guideline_id)
    return JSONResponse(json.loads((d / "manifest.json").read_text()))


@app.get("/api/guidelines/{guideline_id}/pages/{pdf_index}")
def get_page(guideline_id: str, pdf_index: int) -> JSONResponse:
    d = _guideline_dir(guideline_id)
    record_path = d / "pages" / f"p{pdf_index:03d}.json"
    if not record_path.is_file():
        raise HTTPException(status_code=404, detail="page not found")
    record = json.loads(record_path.read_text())
    record["prose_html"] = md.markdown(
        record.get("prose", ""), extensions=["tables", "fenced_code"]
    )
    return JSONResponse(record)


@app.get("/api/guidelines/{guideline_id}/image/{pdf_index}")
def get_image(guideline_id: str, pdf_index: int) -> FileResponse:
    d = _guideline_dir(guideline_id)
    image_path = d / "pages" / f"p{pdf_index:03d}.png"
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(image_path, media_type="image/png")


@app.post("/api/extract")
def run_extract(
    file: UploadFile,
    guideline_id: str = Form(...),
    guideline_title: str = Form(...),
    jurisdiction: str = Form(""),
    version: str = Form(""),
    limit: int = Form(0),
    concurrency: int = Form(25),
) -> StreamingResponse:
    root = output_root()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file.file.read())
        pdf_path = tmp.name

    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            manifest, flags = extract(
                pdf_path,
                str(root / guideline_id),
                guideline_id,
                guideline_title=guideline_title,
                jurisdiction=jurisdiction or None,
                version=version or None,
                limit=limit or None,
                concurrency=concurrency,
                on_page=lambda done, total: events.put(
                    {"type": "progress", "done": done, "total": total}
                ),
            )
            events.put(
                {
                    "type": "done",
                    "guideline_id": guideline_id,
                    "page_count": manifest.page_count,
                    "flags": flags,
                }
            )
        except Exception as exc:  # surface auth/model errors as a stream event
            events.put({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})
        finally:
            os.unlink(pdf_path)
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            item = events.get()
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # no-store so the browser never runs a stale copy of the page's JS
    return HTMLResponse(INDEX_HTML, headers={"Cache-Control": "no-store"})


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Guideline Extractor</title>
<style>
  :root { --line:#d9d9d9; --muted:#666; --bg:#fff; --panel:#fafafa; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 -apple-system,Helvetica,Arial,sans-serif; color:#111; }
  header { border-bottom:1px solid var(--line); padding:10px 14px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  header select, header input, header button { font:13px inherit; padding:4px 6px; border:1px solid var(--line); background:#fff; }
  header button { cursor:pointer; }
  .fld { color:var(--muted); font-size:12px; display:inline-flex; gap:5px; align-items:center; }
  #upload { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  #meta { color:var(--muted); font-size:12px; }
  main { display:grid; grid-template-columns:280px 1fr; height:calc(100vh - 47px); }
  #list { border-right:1px solid var(--line); overflow:auto; }
  #list div { padding:7px 12px; border-bottom:1px solid #eee; cursor:pointer; }
  #list div:hover { background:var(--panel); }
  #list div.sel { background:#eef; }
  #list .pn { color:var(--muted); font-variant-numeric:tabular-nums; margin-right:8px; }
  #block { overflow:auto; padding:0; }
  #blockhead { padding:10px 16px; border-bottom:1px solid var(--line); }
  #blockhead .fields { color:var(--muted); font-size:12px; font-family:ui-monospace,Menlo,monospace; }
  .tabs { display:flex; gap:0; border-bottom:1px solid var(--line); }
  .tabs button { border:none; border-right:1px solid var(--line); background:#fff; padding:8px 16px; cursor:pointer; font:13px inherit; }
  .tabs button.on { background:#111; color:#fff; }
  .pane { padding:16px; display:none; }
  .pane.on { display:block; }
  .prose { max-width:820px; }
  .prose table { border-collapse:collapse; }
  .prose th, .prose td { border:1px solid var(--line); padding:4px 8px; }
  pre.raw { white-space:pre-wrap; font-family:ui-monospace,Menlo,monospace; font-size:12.5px; background:var(--panel); border:1px solid var(--line); padding:12px; }
  img.page { max-width:100%; border:1px solid var(--line); }
  .empty { color:var(--muted); padding:24px; }
  #upload { margin-left:auto; }
  #status { color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<header>
  <strong>Guideline Extractor</strong>
  <label class="fld">View <select id="gsel"></select></label>
  <span id="meta"></span>
  <form id="upload">
    <label class="fld">PDF <input type="file" id="file" accept="application/pdf" required></label>
    <label class="fld">ID <input type="text" id="gid" required size="12"></label>
    <label class="fld">Title <input type="text" id="gtitle" required size="16"></label>
    <label class="fld">Limit <input type="number" id="limit" size="4" title="first N pages; blank = all"></label>
    <button type="submit">Extract</button>
    <span id="status"></span>
  </form>
</header>
<main>
  <div id="list"></div>
  <div id="block"><div class="empty">Select a page.</div></div>
</main>
<script>
const $ = s => document.querySelector(s);
let current = null;

async function loadGuidelines(select) {
  const gs = await (await fetch('/api/guidelines')).json();
  const sel = $('#gsel');
  sel.innerHTML = '';
  for (const g of gs) {
    const o = document.createElement('option');
    o.value = g.guideline_id;
    o.textContent = g.guideline_id + '  (' + g.page_count + ')';
    sel.appendChild(o);
  }
  if (gs.length) { sel.value = select || gs[0].guideline_id; await loadManifest(sel.value); }
  else {
    const o = document.createElement('option'); o.textContent = '(none yet)'; o.disabled = true; sel.appendChild(o);
    $('#list').innerHTML = '<div class="empty">No extractions yet. Upload a PDF above.</div>';
    $('#meta').textContent = '';
  }
}

async function loadManifest(gid) {
  current = gid;
  const m = await (await fetch('/api/guidelines/'+gid+'/manifest')).json();
  $('#meta').textContent = [m.title, m.jurisdiction, m.version, m.page_count+' pages'].filter(Boolean).join(' · ');
  const list = $('#list'); list.innerHTML = '';
  for (const p of m.pages) {
    const d = document.createElement('div');
    d.dataset.idx = p.pdf_index;
    d.innerHTML = '<span class="pn">'+p.page_number+'</span>'+(p.title||'');
    d.onclick = () => selectPage(gid, p.pdf_index, d);
    list.appendChild(d);
  }
}

async function selectPage(gid, idx, el) {
  document.querySelectorAll('#list div').forEach(d => d.classList.remove('sel'));
  if (el) el.classList.add('sel');
  const r = await (await fetch('/api/guidelines/'+gid+'/pages/'+idx)).json();
  const fields = 'guideline_id='+r.guideline_id+'  page_number='+r.page_number+'  pdf_index='+r.pdf_index;
  const recordForJson = Object.assign({}, r); delete recordForJson.prose_html;
  $('#block').innerHTML =
    '<div id="blockhead"><div><strong>'+(r.title||'')+'</strong></div>'
      + '<div class="fields">'+fields+'</div></div>'
    + '<div class="tabs">'
      + '<button data-p="prose" class="on">Prose</button>'
      + '<button data-p="raw">Raw text</button>'
      + '<button data-p="json">JSON</button>'
      + '<button data-p="img">Page image</button>'
    + '</div>'
    + '<div class="pane on" data-p="prose"><div class="prose">'+r.prose_html+'</div></div>'
    + '<div class="pane" data-p="raw"><pre class="raw">'+escapeHtml(r.raw_text||'')+'</pre></div>'
    + '<div class="pane" data-p="json"><pre class="raw">'+escapeHtml(JSON.stringify(recordForJson, null, 2))+'</pre></div>'
    + '<div class="pane" data-p="img"><img class="page" src="/api/guidelines/'+gid+'/image/'+idx+'"></div>';
  document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
    document.querySelectorAll('.tabs button').forEach(x => x.classList.remove('on'));
    document.querySelectorAll('.pane').forEach(x => x.classList.remove('on'));
    b.classList.add('on');
    document.querySelector('.pane[data-p="'+b.dataset.p+'"]').classList.add('on');
  });
}

function escapeHtml(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

$('#gsel').onchange = e => loadManifest(e.target.value);

$('#upload').onsubmit = async e => {
  e.preventDefault();
  const fd = new FormData();
  fd.append('file', $('#file').files[0]);
  fd.append('guideline_id', $('#gid').value);
  fd.append('guideline_title', $('#gtitle').value);
  if ($('#limit').value) fd.append('limit', $('#limit').value);
  $('#status').textContent = 'starting...';
  const res = await fetch('/api/extract', {method:'POST', body:fd});
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream:true});
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const m = JSON.parse(line);
      if (m.type === 'progress') {
        $('#status').textContent = 'processing ' + m.done + ' / ' + m.total;
      } else if (m.type === 'done') {
        $('#status').textContent = 'done: ' + m.page_count + ' pages'
          + (m.flags.length ? (' · QC flags ' + JSON.stringify(m.flags)) : '');
        await loadGuidelines(m.guideline_id);
      } else if (m.type === 'error') {
        $('#status').textContent = 'error: ' + m.detail;
      }
    }
  }
};

loadGuidelines();
</script>
</body>
</html>"""


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
