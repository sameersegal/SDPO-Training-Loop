#!/usr/bin/env python3
"""Read the rendered paper summaries (markdown + math) and annotate them.

Serves the `summary_*.md` reports in a knowledge directory as formatted HTML
(markdown via marked.js, equations via KaTeX — both from CDN, so the viewing
browser needs internet). Select text in a report and save it as an annotation;
each annotation is appended to `<dir>/annotations.json` with a copy of the
highlighted text and a deep link back to that spot in the report.

Usage
-----
    python src/knowledge_reports.py knowledge --serve --port 7500
    # local-only instead of reachable remotely
    python src/knowledge_reports.py knowledge --serve --host 127.0.0.1

annotations.json schema: a JSON list of objects:
    {id, report, title, text, note, link, url, created}

Filenames are validated (summary_*.md only) and confined to the served dir.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import threading
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

REPORT_RE = re.compile(r"^summary_[A-Za-z0-9_.\-]+\.md$")
IMAGE_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.(png|jpg|jpeg|gif|webp|svg)$", re.I)
IMAGE_CT = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "webp": "image/webp", "svg": "image/svg+xml"}
_lock = threading.Lock()


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.annot_path = root / "annotations.json"
        self.images_dir = root / "images"

    def safe_image(self, name: str) -> Path | None:
        if not name or not IMAGE_RE.match(name):
            return None
        p = (self.images_dir / name).resolve()
        if p.parent != self.images_dir.resolve() or not p.is_file():
            return None
        return p

    # --- reports -------------------------------------------------------
    def list_reports(self):
        out = []
        for p in sorted(self.root.glob("summary_*.md")):
            out.append({"file": p.name, "title": self._title(p)})
        return out

    @staticmethod
    def _title(p: Path) -> str:
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith("# "):
                    return line[2:].strip()
        except OSError:
            pass
        return p.stem

    def safe_report(self, name: str) -> Path | None:
        if not name or not REPORT_RE.match(name):
            return None
        p = (self.root / name).resolve()
        if p.parent != self.root.resolve() or not p.is_file():
            return None
        return p

    # --- annotations ---------------------------------------------------
    def load_annots(self):
        if not self.annot_path.exists():
            return []
        try:
            return json.loads(self.annot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _write_annots(self, data):
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".annot-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.annot_path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def add_annot(self, report, text, note, host):
        text = (text or "").strip()
        if not REPORT_RE.match(report or ""):
            raise ValueError("invalid report")
        if not text:
            raise ValueError("empty highlight")
        title = self._title(self.root / report)
        path = f"/?report={quote(report)}&hl={quote(text[:200])}"
        url = f"http://{host}{path}" if host else path
        entry = {
            "id": uuid.uuid4().hex[:8],
            "report": report,
            "title": title,
            "text": text,
            "note": (note or "").strip(),
            "link": path,
            "url": url,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        with _lock:
            data = self.load_annots()
            data.append(entry)
            self._write_annots(data)
        return entry

    def delete_annot(self, aid):
        with _lock:
            data = self.load_annots()
            n = len(data)
            data = [a for a in data if a.get("id") != aid]
            if len(data) != n:
                self._write_annots(data)
            return len(data) != n


PAGE = r'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDPO summaries — read &amp; annotate</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
  :root { color-scheme: light dark; --fg:#1f2328; --bg:#f6f8fa; --card:#fff; --bd:#d0d7de; --mut:#57606a; --acc:#1a7f37; }
  @media (prefers-color-scheme: dark){ :root{ --fg:#e6edf3; --bg:#0d1117; --card:#161b22; --bd:#30363d; --mut:#8b949e; --acc:#3fb950; } }
  * { box-sizing: border-box; }
  html, body { width:100%; max-width:100%; overflow-x:hidden; }
  body { margin:0; font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; color:var(--fg); background:var(--bg); }
  .layout { display:grid; grid-template-columns:240px minmax(0,1fr) 300px; min-height:100vh; width:100%; max-width:100%; }
  nav, aside { border-color:var(--bd); background:var(--card); }
  nav { border-right:1px solid var(--bd); padding:14px 10px; position:sticky; top:0; height:100vh; overflow:auto; }
  aside { border-left:1px solid var(--bd); padding:14px 12px; position:sticky; top:0; height:100vh; overflow:auto; }
  nav h2, aside h2 { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--mut); margin:0 0 8px; }
  .rlink { display:block; padding:6px 8px; border-radius:6px; cursor:pointer; font-size:13px; color:inherit; text-decoration:none; }
  .rlink:hover { background:var(--bg); }
  .rlink.active { background:var(--acc); color:#fff; font-weight:600; }
  main { padding:26px 38px 80px; max-width:880px; min-width:0; width:100%; margin:0 auto; }
  .content { font-size:15.5px; min-width:0; max-width:100%; overflow-wrap:break-word; }
  .content > * { max-width:100%; }
  .content p, .content li { overflow-wrap:anywhere; }
  .content img, .content svg, .content canvas, .content video { max-width:100%; height:auto; }
  .content h1 { font-size:24px; margin:.2em 0 .6em; }
  .content h2 { font-size:19px; margin:1.4em 0 .5em; border-bottom:1px solid var(--bd); padding-bottom:.2em; }
  .content h3 { font-size:16px; margin:1.2em 0 .4em; }
  .content code { background:rgba(127,127,127,.16); padding:.12em .35em; border-radius:5px; font-size:.9em; }
  .content pre { background:rgba(127,127,127,.12); padding:12px 14px; border-radius:8px; overflow:auto; max-width:100%; }
  .content pre code { background:none; padding:0; }
  .content table { border-collapse:collapse; margin:1em 0; display:block; overflow:auto; max-width:100%; }
  .content th, .content td { border:1px solid var(--bd); padding:6px 10px; text-align:left; }
  .content blockquote { border-left:3px solid var(--bd); margin:1em 0; padding:.2em 1em; color:var(--mut); }
  .content a { color:#3b82f6; }
  .content img { display:block; margin:14px auto 4px; border:1px solid var(--bd); border-radius:6px; background:#fff; }
  .content p:has(> img) { text-align:center; }
  .content em { color:var(--mut); }
  .content .katex-display { max-width:100%; overflow-x:auto; overflow-y:hidden; padding-bottom:2px; }
  .content .katex-display > .katex { max-width:100%; }
  mark.hl { background:#fff3b0; color:inherit; border-radius:2px; padding:0 1px; }
  @media (prefers-color-scheme: dark){ mark.hl { background:#5c4d00; } }
  mark.hl.flash { animation: flash 1.6s ease; }
  @keyframes flash { 0%,40%{ background:#ffd33d; } 100%{ background:#fff3b0; } }
  #savebtn { position:absolute; z-index:50; display:none; background:var(--acc); color:#fff; border:none;
    border-radius:6px; padding:6px 10px; font-size:13px; cursor:pointer; box-shadow:0 2px 8px rgba(0,0,0,.25); }
  .annot { border:1px solid var(--bd); border-radius:8px; padding:8px 10px; margin:0 0 10px; font-size:13px; background:var(--bg); }
  .annot .q { border-left:3px solid var(--acc); padding-left:8px; margin:0 0 6px; color:var(--fg); }
  .annot .note { color:var(--mut); font-style:italic; margin:0 0 6px; }
  .annot .row { display:flex; gap:8px; }
  .annot button { background:none; border:1px solid var(--bd); border-radius:5px; padding:2px 7px; font-size:12px; cursor:pointer; color:inherit; }
  .annot button:hover { border-color:var(--acc); }
  .empty { color:var(--mut); font-size:13px; }
  .hint { color:var(--mut); font-size:12px; margin:0 0 12px; }
  /* mobile top bar + off-canvas drawers (hidden on desktop) */
  .topbar { display:none; }
  .backdrop { display:none; }
  @media (max-width: 1100px){
    .layout { grid-template-columns:minmax(0,1fr); }
    .topbar { display:flex; gap:8px; align-items:center; position:sticky; top:0; z-index:65;
      background:var(--card); border-bottom:1px solid var(--bd); padding:8px 10px; }
    .topbar button { background:none; border:1px solid var(--bd); border-radius:6px; color:inherit;
      padding:6px 11px; font-size:14px; cursor:pointer; white-space:nowrap; }
    .topbar .ttl { flex:1; min-width:0; text-align:center; font-size:13px; color:var(--mut);
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    nav, aside { position:fixed; top:0; bottom:0; height:auto; width:82%; max-width:320px; z-index:70;
      transition:transform .22s ease; box-shadow:0 0 26px rgba(0,0,0,.35); }
    nav { left:0; transform:translateX(-100%); border-right:1px solid var(--bd); }
    aside { right:0; transform:translateX(100%); border-left:1px solid var(--bd); }
    body.nav-open nav { transform:translateX(0); }
    body.aside-open aside { transform:translateX(0); }
    body.nav-open .backdrop, body.aside-open .backdrop { display:block; }
    .backdrop { position:fixed; inset:0; background:rgba(0,0,0,.45); z-index:68; }
    main { padding:18px 14px 70px; max-width:none; }
    .content { font-size:15px; }
    .content p .katex, .content li .katex { white-space:normal; }
  }
</style></head>
<body>
<div class="topbar">
  <button id="navtoggle" aria-label="Toggle reports">☰ Reports</button>
  <span class="ttl" id="topttl"></span>
  <button id="asidetoggle" aria-label="Toggle annotations">✎ Notes</button>
</div>
<div class="backdrop" id="backdrop"></div>
<div class="layout">
  <nav><h2>Reports</h2><div id="reports"></div></nav>
  <main>
    <p class="hint">Select any text in a report to save a highlight. Saved highlights persist and deep-link back here.</p>
    <div id="content" class="content"></div>
  </main>
  <aside>
    <h2>Annotations <span id="acount"></span></h2>
    <div id="annots"></div>
  </aside>
</div>
<button id="savebtn">★ Save highlight</button>
<script>
const reportsEl = document.getElementById("reports");
const contentEl = document.getElementById("content");
const annotsEl  = document.getElementById("annots");
const acountEl  = document.getElementById("acount");
const saveBtn   = document.getElementById("savebtn");
let current = null;          // current report filename
let katexReady = false;

const esc = s => (s||"").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

// --- markdown + math -------------------------------------------------
function renderMarkdown(md){
  const math=[];
  const NUL="\uE000";
  const stash=(tex,display)=>{ math.push({tex,display}); return NUL+"M"+(math.length-1)+NUL; };
  // pull code out FIRST so a $ inside code (e.g. a regex `...proj$`) is never seen as math
  const code=[];
  const stashCode=(s)=>{ code.push(s); return NUL+"C"+(code.length-1)+NUL; };
  md = md.replace(/```[\s\S]*?```/g, stashCode);   // fenced blocks
  md = md.replace(/~~~[\s\S]*?~~~/g, stashCode);
  md = md.replace(/`[^`\n]*`/g, stashCode);        // inline code
  md = md.replace(/\\\$/g, NUL+"D"+NUL);                       // escaped dollar
  const usd=[];
  md = md.replace(/\$(\d[\d.,%kKmMxX]*)/g, (m,g)=>{ usd.push("$"+g); return NUL+"U"+(usd.length-1)+NUL; }); // currency
  md = md.replace(/\$\$([\s\S]+?)\$\$/g, (m,t)=>stash(t,true));
  md = md.replace(/\\\[([\s\S]+?)\\\]/g, (m,t)=>stash(t,true));
  md = md.replace(/\\\(([\s\S]+?)\\\)/g, (m,t)=>stash(t,false));
  md = md.replace(/\$([^\n$]+?)\$/g, (m,t)=>stash(t,false));
  // a lone "~" means "approximately" here, not GFM strikethrough — escape it (keep real ~~strike~~ pairs)
  md = md.replace(/(?<!~)~(?!~)/g, "\\~");
  md = md.replace(new RegExp(NUL+"C(\\d+)"+NUL,"g"), (m,i)=>code[+i]);  // restore code before marked
  let html = marked.parse(md, {gfm:true, breaks:false});
  html = html.replace(new RegExp(NUL+"M(\\d+)"+NUL,"g"), (m,i)=>{
    const {tex,display}=math[+i];
    if(!katexReady || !window.katex) return '<code>'+esc(tex)+'</code>';
    try { return katex.renderToString(tex,{displayMode:display,throwOnError:false}); }
    catch(e){ return '<code>'+esc(tex)+'</code>'; }
  });
  html = html.replace(new RegExp(NUL+"U(\\d+)"+NUL,"g"), (m,i)=>esc(usd[+i]));
  html = html.replace(new RegExp(NUL+"D"+NUL,"g"), "$");
  return html;
}

// --- highlight helpers ----------------------------------------------
function markText(text, id, flash){
  if(!text) return null;
  const needle = text.trim();
  const tw = document.createTreeWalker(contentEl, NodeFilter.SHOW_TEXT);
  let n;
  while((n=tw.nextNode())){
    const idx = n.nodeValue.indexOf(needle);
    if(idx>=0){
      const r = document.createRange();
      r.setStart(n, idx); r.setEnd(n, idx+needle.length);
      const mk = document.createElement("mark");
      mk.className = "hl" + (flash?" flash":"");
      if(id) mk.dataset.id = id;
      try { r.surroundContents(mk); return mk; } catch(e){ return null; }
    }
  }
  return null;
}

// --- data loading ----------------------------------------------------
async function loadReports(){
  const reports = await (await fetch("/api/reports")).json();
  reportsEl.innerHTML = reports.map(r =>
    `<a class="rlink" data-file="${esc(r.file)}" title="${esc(r.file)}">${esc(r.title)}</a>`).join("");
  reportsEl.querySelectorAll(".rlink").forEach(a =>
    a.addEventListener("click", () => openReport(a.dataset.file)));
  return reports;
}

async function openReport(file, hl){
  current = file;
  reportsEl.querySelectorAll(".rlink").forEach(a => a.classList.toggle("active", a.dataset.file===file));
  const act = reportsEl.querySelector('.rlink.active');
  document.getElementById("topttl").textContent = act ? act.textContent : file;
  document.body.classList.remove("nav-open");          // close the drawer after picking a report (mobile)
  const md = await (await fetch("/api/report?file="+encodeURIComponent(file))).text();
  contentEl.innerHTML = renderMarkdown(md);
  history.replaceState(null, "", "/?report="+encodeURIComponent(file));
  await renderAnnots();           // also re-applies saved highlights
  if(hl){
    const mk = markText(hl, null, true);
    if(mk) mk.scrollIntoView({behavior:"smooth", block:"center"});
  }
}

async function renderAnnots(){
  const all = await (await fetch("/api/annotations?report="+encodeURIComponent(current||"")).then(r=>r.json()));
  acountEl.textContent = all.length ? "("+all.length+")" : "";
  annotsEl.innerHTML = all.length
    ? all.map(a => `<div class="annot" data-id="${esc(a.id)}">
        <p class="q">${esc(a.text)}</p>
        ${a.note?`<p class="note">${esc(a.note)}</p>`:""}
        <div class="row">
          <button data-act="jump" data-id="${esc(a.id)}">Jump</button>
          <button data-act="copy" data-link="${esc(a.url||a.link)}">Copy link</button>
          <button data-act="del" data-id="${esc(a.id)}">Delete</button>
        </div></div>`).join("")
    : `<p class="empty">No annotations yet for this report.</p>`;
  // re-apply persistent highlights
  all.forEach(a => { if(!contentEl.querySelector('mark[data-id="'+a.id+'"]')) markText(a.text, a.id, false); });
}

// --- annotation actions ---------------------------------------------
async function saveHighlight(text){
  const note = prompt("Optional note for this highlight:", "") ?? "";
  try {
    const a = await (await fetch("/api/annotate", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({report:current, text, note})})).json();
    if(a.error) throw new Error(a.error);
    await renderAnnots();
    const mk = contentEl.querySelector('mark[data-id="'+a.id+'"]');
    if(mk){ mk.classList.add("flash"); mk.scrollIntoView({behavior:"smooth", block:"center"}); }
  } catch(e){ alert("Save failed: "+e.message); }
}

annotsEl.addEventListener("click", async e => {
  const b = e.target.closest("button[data-act]"); if(!b) return;
  if(b.dataset.act==="jump"){
    document.body.classList.remove("aside-open");       // close the drawer so the jump is visible (mobile)
    const mk = contentEl.querySelector('mark[data-id="'+b.dataset.id+'"]');
    if(mk){ mk.classList.remove("flash"); void mk.offsetWidth; mk.classList.add("flash"); mk.scrollIntoView({behavior:"smooth", block:"center"}); }
  } else if(b.dataset.act==="copy"){
    const link = new URL(b.dataset.link, location.origin).href;
    navigator.clipboard.writeText(link).then(()=>{ b.textContent="Copied ✓"; setTimeout(()=>b.textContent="Copy link",1200); });
  } else if(b.dataset.act==="del"){
    await fetch("/api/annotate/delete", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({id:b.dataset.id})});
    const mk = contentEl.querySelector('mark[data-id="'+b.dataset.id+'"]');
    if(mk){ const t=document.createTextNode(mk.textContent); mk.replaceWith(t); }
    renderAnnots();
  }
});

// --- selection -> floating save button (mouse + touch) --------------
function onSelect(){
  const sel = window.getSelection();
  const text = sel && sel.toString();
  if(text && text.trim().length>=4 && contentEl.contains(sel.anchorNode) && contentEl.contains(sel.focusNode)){
    const r = sel.getRangeAt(0).getBoundingClientRect();
    saveBtn.style.left = Math.min(window.scrollX + r.left, window.scrollX + window.innerWidth - 150) + "px";
    saveBtn.style.top  = (window.scrollY + r.bottom + 6) + "px";
    saveBtn.style.display = "block";
    saveBtn.onclick = () => { saveBtn.style.display="none"; saveHighlight(text.trim()); };
  } else if(!saveBtn.contains(document.activeElement)) {
    saveBtn.style.display = "none";
  }
}
document.addEventListener("mouseup", onSelect);
document.addEventListener("touchend", onSelect);

// --- mobile drawers -------------------------------------------------
document.getElementById("navtoggle").addEventListener("click", () => {
  document.body.classList.toggle("nav-open"); document.body.classList.remove("aside-open");
});
document.getElementById("asidetoggle").addEventListener("click", () => {
  document.body.classList.toggle("aside-open"); document.body.classList.remove("nav-open");
});
document.getElementById("backdrop").addEventListener("click", () => document.body.classList.remove("nav-open","aside-open"));
document.addEventListener("keydown", e => { if(e.key==="Escape") document.body.classList.remove("nav-open","aside-open"); });

// --- boot ------------------------------------------------------------
function whenKatex(cb){
  if(window.katex){ katexReady=true; cb(); return; }
  let t=setInterval(()=>{ if(window.katex){ katexReady=true; clearInterval(t); cb(); } }, 60);
  setTimeout(()=>clearInterval(t), 4000);
}

(async function(){
  const reports = await loadReports();
  const q = new URLSearchParams(location.search);
  const want = q.get("report");
  const hl = q.get("hl");
  const first = (reports.find(r=>r.file===want) || reports[0] || {}).file;
  whenKatex(() => { if(first) openReport(first, hl); });
})();
</script>
</body></html>
'''


def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                self._send(200, PAGE)
            elif u.path == "/api/reports":
                self._json(store.list_reports())
            elif u.path == "/api/report":
                p = store.safe_report((q.get("file") or [""])[0])
                if not p:
                    self._send(404, "not found", "text/plain; charset=utf-8")
                else:
                    self._send(200, p.read_text(encoding="utf-8"), "text/markdown; charset=utf-8")
            elif u.path == "/api/annotations":
                rep = (q.get("report") or [""])[0]
                data = store.load_annots()
                if rep:
                    data = [a for a in data if a.get("report") == rep]
                self._json(data)
            elif u.path.startswith("/images/"):
                p = store.safe_image(u.path[len("/images/"):])
                if not p:
                    self._send(404, "not found", "text/plain; charset=utf-8")
                else:
                    ct = IMAGE_CT.get(p.suffix.lower().lstrip("."), "application/octet-stream")
                    self._send(200, p.read_bytes(), ct)
            else:
                self._send(404, "not found", "text/plain; charset=utf-8")

        def do_POST(self):
            u = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                self._json({"error": f"bad request: {e}"}, 400)
                return
            if u.path == "/api/annotate":
                try:
                    entry = store.add_annot(req.get("report"), req.get("text"),
                                            req.get("note"), self.headers.get("Host"))
                    self._json(entry)
                except ValueError as e:
                    self._json({"error": str(e)}, 400)
            elif u.path == "/api/annotate/delete":
                store.delete_annot(req.get("id"))
                self._json({"ok": True})
            else:
                self._send(404, "not found", "text/plain; charset=utf-8")

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?", default=Path("knowledge"),
                    help="directory holding summary_*.md (default: knowledge)")
    ap.add_argument("--serve", action="store_true", help="(default) run the server")
    ap.add_argument("--host", default="0.0.0.0", help="bind host (default 0.0.0.0)")
    ap.add_argument("--port", type=int, default=7500, help="port (default 7500)")
    args = ap.parse_args()

    root = args.dir.resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    store = Store(root)
    reports = store.list_reports()
    if not reports:
        sys.exit(f"No summary_*.md files in {root}")
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Serving {len(reports)} reports from {root}")
    print(f"Annotations → {store.annot_path}")
    print(f"Open http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
