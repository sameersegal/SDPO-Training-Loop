#!/usr/bin/env python3
"""View — and reclassify — the SDPO papers CSV in one web interface.

The CSV is expected to have the columns written by knowledge/citing_papers.csv:
    arxiv_id, relationship, related, title, url, abstract
Extra columns are ignored; missing ones degrade gracefully.

Usage
-----
    # serve an editable view (click a button to change a paper's classification;
    # the change is written straight back to the CSV)
    python src/knowledge_view.py knowledge/citing_papers.csv --serve --port 7400

    # local-only instead of reachable remotely
    python src/knowledge_view.py knowledge/citing_papers.csv --serve --host 127.0.0.1

    # write a static, read-only HTML file (good for scp/offline; no save buttons)
    python src/knowledge_view.py knowledge/citing_papers.csv --out knowledge/papers.html

Each save rewrites the whole CSV atomically (temp file + os.replace), preserving
all other columns, row order, and quoting. Single editor at a time.
"""
import argparse
import csv
import html
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Allowed values for the `related` column, in display order.
CLASSES = ["original", "yes", "tangential", "no", ""]
LABEL = {"original": "Original", "yes": "Related", "tangential": "Tangential",
         "no": "Not related", "": "Unclassified"}
COLOR = {"original": "#6f42c1", "yes": "#1a7f37", "tangential": "#9a6700",
         "no": "#8c959f", "": "#8c959f"}

_lock = threading.Lock()


class Store:
    """In-memory authoritative copy of the CSV; rewrites the file on change."""

    def __init__(self, path: Path):
        with path.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            self.fieldnames = r.fieldnames or []
            self.rows = list(r)
        self.path = path
        if "arxiv_id" not in self.fieldnames or "related" not in self.fieldnames:
            sys.exit("CSV must have at least 'arxiv_id' and 'related' columns.")
        if not self.rows:
            sys.exit(f"No rows in {path}")
        self._ensure_reviewed_col()

    def _ensure_reviewed_col(self):
        if "reviewed" not in self.fieldnames:
            i = self.fieldnames.index("related") + 1 if "related" in self.fieldnames else len(self.fieldnames)
            self.fieldnames.insert(i, "reviewed")
        for r in self.rows:
            if not r.get("reviewed"):
                r["reviewed"] = "no"

    def set_related(self, arxiv_id: str, related: str) -> bool:
        """Set classification; changing it auto-marks the paper reviewed."""
        if related not in CLASSES:
            raise ValueError(f"invalid class: {related!r}")
        with _lock:
            hit = next((r for r in self.rows if r.get("arxiv_id") == arxiv_id), None)
            if hit is None:
                return False
            hit["related"] = related
            hit["reviewed"] = "yes"
            self._write()
            return True

    def set_reviewed(self, arxiv_id: str, reviewed: bool) -> bool:
        with _lock:
            hit = next((r for r in self.rows if r.get("arxiv_id") == arxiv_id), None)
            if hit is None:
                return False
            hit["reviewed"] = "yes" if reviewed else "no"
            self._write()
            return True

    def _write(self):
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".papers-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames, quoting=csv.QUOTE_MINIMAL)
                w.writeheader()
                w.writerows(self.rows)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


def page_html(rows, title: str, *, editable: bool, csv_name: str = "") -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    safe_title = html.escape(title)
    hint = (f'<p class="hint">Click a classification button to save it straight to '
            f'<code>{html.escape(csv_name)}</code>.</p>' if editable else "")
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0; background: #f6f8fa; color: #1f2328; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #0d1117; color: #e6edf3; }}
    .card, header {{ background: #161b22 !important; border-color: #30363d !important; }}
    input, .seg button, .chip {{ background: #0d1117 !important; color: #e6edf3 !important; border-color: #30363d !important; }}
    .abs {{ color: #c9d1d9 !important; }} a {{ color: #6cb6ff; }}
  }}
  header {{ position: sticky; top: 0; z-index: 5; background: #fff; border-bottom: 1px solid #d0d7de; padding: 14px 20px; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  .hint {{ color: #57606a; font-size: 12.5px; margin: 0 0 10px; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
  input[type=search] {{ flex: 1 1 240px; min-width: 180px; padding: 7px 10px; font-size: 14px;
    border: 1px solid #d0d7de; border-radius: 6px; background: #fff; color: inherit; }}
  .chip {{ cursor: pointer; border: 1px solid #d0d7de; background: #fff; color: inherit;
    border-radius: 999px; padding: 5px 12px; font-size: 13px; --c: #57606a; }}
  .chip.active {{ border-color: var(--c); box-shadow: inset 0 0 0 1px var(--c); font-weight: 600; }}
  .chip .n {{ opacity: .6; font-variant-numeric: tabular-nums; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 18px 20px 60px; }}
  .meta {{ color: #57606a; font-size: 13px; margin: 0 0 14px; }}
  .card {{ background: #fff; border: 1px solid #d0d7de; border-radius: 10px; padding: 14px 16px; margin: 0 0 12px; }}
  .row1 {{ display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }}
  .title {{ font-size: 16px; font-weight: 600; margin: 0; }}
  .title a {{ text-decoration: none; }} .title a:hover {{ text-decoration: underline; }}
  .badge {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .03em;
    color: #fff; padding: 2px 8px; border-radius: 999px; white-space: nowrap; }}
  .sub {{ color: #57606a; font-size: 12.5px; margin: 4px 0 8px; font-variant-numeric: tabular-nums; }}
  .sub a {{ color: inherit; }}
  .seg {{ display: inline-flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 10px; align-items: center; }}
  .seg button {{ cursor: pointer; border: 1px solid #d0d7de; background: #fff; color: inherit;
    border-radius: 6px; padding: 4px 10px; font-size: 12.5px; --c: #57606a; }}
  .seg button.on {{ color: #fff; background: var(--c); border-color: var(--c); font-weight: 600; }}
  .saved {{ font-size: 12px; color: #1a7f37; margin-left: 4px; opacity: 0; transition: opacity .15s; }}
  .saved.show {{ opacity: 1; }} .saved.err {{ color: #cf222e; }}
  .seg .sep {{ width: 1px; align-self: stretch; background: #d0d7de; margin: 0 2px; }}
  .revbtn {{ cursor: pointer; border: 1px solid #d0d7de; background: #fff; color: inherit;
    border-radius: 6px; padding: 4px 10px; font-size: 12.5px; }}
  .revbtn.on {{ background: #1a7f37; color: #fff; border-color: #1a7f37; font-weight: 600; }}
  .card.reviewed {{ border-left: 3px solid #1a7f37; }}
  .abs {{ color: #313840; font-size: 14px; margin: 0; white-space: pre-wrap; }}
  .abs.clamp {{ display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; }}
  body.expandall .abs.clamp {{ display: block; -webkit-line-clamp: unset; overflow: visible; }}
  body.expandall .toggle {{ display: none; }}
  .toggle {{ margin-top: 6px; font-size: 12.5px; cursor: pointer; color: #57606a; background: none; border: none; padding: 0; }}
  .toggle:hover {{ text-decoration: underline; }}
  .empty {{ color: #57606a; padding: 30px; text-align: center; }}
</style></head><body>
<header>
  <h1>{safe_title}</h1>
  {hint}
  <div class="controls">
    <input id="q" type="search" placeholder="Filter by title, id, or abstract text…" autofocus>
    <span id="chips"></span>
    <span id="reviewchips"></span>
    <button class="chip" id="expandall">Expand all abstracts</button>
  </div>
</header>
<main><p class="meta" id="count"></p><div id="list"></div></main>
<script>
const DATA = {payload};
const CLASSES = {json.dumps(CLASSES)};
const LABEL = {json.dumps(LABEL)};
const COLOR = {json.dumps(COLOR)};
const EDITABLE = {json.dumps(editable)};
let activeRel = "__all__";
let activeReview = "__all__";       // __all__ | yes | no
const expanded = new Set();         // arxiv_ids whose abstract is expanded (survives re-render)

const esc = s => (s||"").replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c]));
const listEl = document.getElementById("list");
const countEl = document.getElementById("count");
const chipsEl = document.getElementById("chips");
const reviewEl = document.getElementById("reviewchips");
const qEl = document.getElementById("q");
const isReviewed = p => (p.reviewed||"no") === "yes";

function renderChips() {{
  const counts = {{}}; DATA.forEach(p => counts[p.related||""] = (counts[p.related||""]||0)+1);
  const order = CLASSES.filter(c => counts[c]);
  let h = `<button class="chip ${{activeRel==="__all__"?"active":""}}" data-rel="__all__">All <span class="n">${{DATA.length}}</span></button>`;
  order.forEach(c => h += `<button class="chip ${{activeRel===c?"active":""}}" data-rel="${{esc(c)}}" style="--c:${{COLOR[c]}}">${{esc(LABEL[c]||c)}} <span class="n">${{counts[c]}}</span></button>`);
  chipsEl.innerHTML = h;
  chipsEl.querySelectorAll(".chip").forEach(c => c.addEventListener("click", () => {{ activeRel = c.dataset.rel; renderChips(); render(); }}));
}}

function renderReviewChips() {{
  if (!EDITABLE) return;
  const opts = [["__all__","All"],["no","Unreviewed"],["yes","Reviewed"]];
  let h = "";
  opts.forEach(([v,lab]) => {{
    const n = v==="__all__" ? DATA.length : DATA.filter(p => (p.reviewed||"no")===v).length;
    h += `<button class="chip ${{activeReview===v?"active":""}}" data-rev="${{v}}" style="--c:#1a7f37">${{lab}} <span class="n">${{n}}</span></button>`;
  }});
  reviewEl.innerHTML = h;
  reviewEl.querySelectorAll(".chip").forEach(c => c.addEventListener("click", () => {{ activeReview = c.dataset.rev; renderReviewChips(); render(); }}));
}}

function classControl(p) {{
  if (EDITABLE) {{
    const seg = CLASSES.map(c =>
      `<button class="${{(p.related||"")===c?"on":""}}" style="--c:${{COLOR[c]}}" data-act="class" data-id="${{esc(p.arxiv_id)}}" data-c="${{esc(c)}}">${{esc(LABEL[c]||c)}}</button>`
    ).join("");
    const rev = `<button class="revbtn ${{isReviewed(p)?"on":""}}" data-act="review" data-id="${{esc(p.arxiv_id)}}">${{isReviewed(p)?"Reviewed ✓":"Mark reviewed"}}</button>`;
    return `<div class="seg">${{seg}}<span class="sep"></span>${{rev}}<span class="saved" id="saved-${{esc(p.arxiv_id)}}"></span></div>`;
  }}
  const rel = p.related || "";
  return `<span class="badge" style="background:${{COLOR[rel]||"#8c959f"}}">${{esc(LABEL[rel]||rel||"Unclassified")}}</span>`;
}}

function card(p) {{
  const id = p.arxiv_id || "";
  const url = p.url || ("https://arxiv.org/abs/" + id);
  const titleHtml = (p.url||id)
    ? `<a href="${{esc(url)}}" target="_blank" rel="noopener">${{esc(p.title||"(untitled)")}}</a>` : esc(p.title||"(untitled)");
  const abs = p.abstract || ""; const long = abs.length > 360; const open = expanded.has(id);
  const ctrl = classControl(p);
  const head = EDITABLE
    ? `<p class="title">${{titleHtml}}</p>`
    : `<div class="row1">${{ctrl}}<p class="title">${{titleHtml}}</p></div>`;
  const revCls = (EDITABLE && isReviewed(p)) ? "reviewed" : "";
  return `<div class="card ${{revCls}}" data-id="${{esc(id)}}">
    ${{head}}
    <div class="sub">${{esc(id)}}${{p.relationship?" · "+esc(p.relationship):""}}
      ${{id?` · <a href="${{esc(url)}}" target="_blank" rel="noopener">arxiv.org/abs/${{esc(id)}}</a>`:""}}</div>
    ${{EDITABLE ? ctrl : ""}}
    <p class="abs ${{long&&!open?"clamp":""}}" id="abs-${{esc(id)}}">${{esc(abs)||"<em>(no abstract)</em>"}}</p>
    ${{long?`<button class="toggle" data-tog="${{esc(id)}}">${{open?"Show less ▴":"Show more ▾"}}</button>`:""}}
  </div>`;
}}

function render() {{
  const q = qEl.value.trim().toLowerCase();
  const rows = DATA.filter(p => {{
    if (activeRel!=="__all__" && (p.related||"")!==activeRel) return false;
    if (activeReview!=="__all__" && (p.reviewed||"no")!==activeReview) return false;
    if (!q) return true;
    return ((p.title||"")+" "+(p.arxiv_id||"")+" "+(p.abstract||"")).toLowerCase().includes(q);
  }});
  const rev = DATA.filter(isReviewed).length;
  countEl.textContent = EDITABLE
    ? `${{rows.length}} of ${{DATA.length}} shown · ${{rev}}/${{DATA.length}} reviewed`
    : `${{rows.length}} of ${{DATA.length}} paper(s)`;
  listEl.innerHTML = rows.length ? rows.map(card).join("") : `<p class="empty">No papers match.</p>`;
}}

function flash(id, msg, err) {{
  const el = document.getElementById("saved-"+id);
  if (!el) return;
  el.textContent = msg; el.className = "saved show" + (err?" err":"");
  setTimeout(()=>el.classList.remove("show"), 1200);
}}

async function post(url, body) {{
  const res = await fetch(url, {{ method:"POST", headers:{{"Content-Type":"application/json"}}, body: JSON.stringify(body) }});
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}}

async function classify(id, cls) {{
  try {{
    await post("/api/classify", {{arxiv_id:id, related:cls}});
    const p = DATA.find(x => x.arxiv_id===id);
    if (p) {{ p.related = cls; p.reviewed = "yes"; }}   // changing classification auto-marks reviewed
    renderChips(); renderReviewChips(); render();
    flash(id, "saved ✓");
  }} catch (e) {{ flash(id, "save failed", true); }}
}}

async function review(id) {{
  const p = DATA.find(x => x.arxiv_id===id);
  const next = !(p && isReviewed(p));
  try {{
    await post("/api/review", {{arxiv_id:id, reviewed:next}});
    if (p) p.reviewed = next ? "yes" : "no";
    renderReviewChips(); render();
    flash(id, next ? "reviewed ✓" : "unmarked");
  }} catch (e) {{ flash(id, "save failed", true); }}
}}

listEl.addEventListener("click", e => {{
  const b = e.target.closest("button[data-act]");
  if (b) {{ b.dataset.act==="class" ? classify(b.dataset.id, b.dataset.c) : review(b.dataset.id); return; }}
  const tog = e.target.closest(".toggle");
  if (tog) {{ const id = tog.dataset.tog;
    expanded.has(id) ? expanded.delete(id) : expanded.add(id);
    const el = document.getElementById("abs-"+id), open = expanded.has(id);
    el.classList.toggle("clamp", !open); tog.textContent = open?"Show less ▴":"Show more ▾"; }}
}});

const expandBtn = document.getElementById("expandall");
expandBtn.addEventListener("click", () => {{
  const on = document.body.classList.toggle("expandall");
  expandBtn.classList.toggle("active", on);
  expandBtn.textContent = on ? "Collapse all abstracts" : "Expand all abstracts";
}});

qEl.addEventListener("input", render);
renderChips(); renderReviewChips(); render();
</script></body></html>
"""


def make_handler(store: Store, title: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="text/html; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = page_html(store.rows, title, editable=True,
                                 csv_name=store.path.name).encode("utf-8")
                self._send(200, body)
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

        def do_POST(self):
            if self.path not in ("/api/classify", "/api/review"):
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                aid = str(req["arxiv_id"])
                if self.path == "/api/classify":
                    ok = store.set_related(aid, str(req["related"]))
                else:
                    ok = store.set_reviewed(aid, bool(req["reviewed"]))
            except ValueError as e:
                self._send(400, str(e).encode(), "text/plain; charset=utf-8")
                return
            except Exception as e:
                self._send(400, f"bad request: {e}".encode(), "text/plain; charset=utf-8")
                return
            if not ok:
                self._send(404, b"unknown arxiv_id", "text/plain; charset=utf-8")
                return
            self._send(200, json.dumps({"ok": True}).encode(), "application/json")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="papers CSV (e.g. knowledge/citing_papers.csv)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write a static read-only HTML file instead of serving")
    ap.add_argument("--title", default="SDPO literature — papers citing SDPO", help="page title")
    ap.add_argument("--serve", action="store_true", help="serve an editable view")
    ap.add_argument("--host", default="0.0.0.0", help="bind host for --serve (default 0.0.0.0)")
    ap.add_argument("--port", type=int, default=7400, help="port for --serve")
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")
    store = Store(args.csv)

    if args.serve:
        httpd = ThreadingHTTPServer((args.host, args.port), make_handler(store, args.title))
        print(f"Editing {args.csv} ({len(store.rows)} papers)")
        print(f"Serving editable view at http://{args.host}:{args.port}/  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        out = args.out or args.csv.with_name("papers.html")
        out.write_text(page_html(store.rows, args.title, editable=False), encoding="utf-8")
        print(f"Wrote static read-only {out} ({len(store.rows)} papers). "
              f"Use --serve for the editable interface.")


if __name__ == "__main__":
    main()
