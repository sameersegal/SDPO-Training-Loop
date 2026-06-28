#!/usr/bin/env python3
"""Render per-token SDPO advantage A_t over the student's attempt as readable, color-graded
text — so we can verify the critique's signal is localized at the *actual flaw* (in reasoning
OR code), not just summarize it as a code-vs-prose ratio.

For each failure in a teacher_eval.json (A_t arrays + critique) joined with the rollouts
(attempt text), emits one self-contained HTML with: the problem, the critic feedback, the
A_t line plot, and the attempt with each token shaded by A_t
  red  = A_t>0  (teacher MORE confident than student -> "keep this token")
  blue = A_t<0  (teacher LESS confident -> "change / flaw region")
Hover a token for its A_t / teacher_lp / student_lp.

  S=../../src
  PYTHONPATH=$S python $S/plot_advantage_text.py --eval scratch_tadv/teacher_eval.json \
    --rollouts reports/iteration-05/data/qwen3_rollouts.json --out scratch_tadv/advantage_view.html
"""
import argparse
import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from sdpo_ojbench import PROMPT_BY_ID, CPP_PROMPT_BY_ID


def color(a, vmax):
    t = max(-1.0, min(1.0, a / vmax)) if vmax else 0.0
    if t >= 0:
        return f"rgba(220,50,47,{t*0.85:.3f})"   # red: keep (A_t>0)
    return f"rgba(38,139,210,{-t*0.85:.3f})"     # blue: change (A_t<0)


def at_plot_b64(a_t, fence):
    a = np.asarray(a_t, dtype=float)
    x = np.arange(len(a))
    fig, ax = plt.subplots(figsize=(11, 1.8))
    # match the text shading: red fill where A_t>0 ("keep"), blue where A_t<0 ("change/flaw")
    ax.fill_between(x, 0, a, where=a >= 0, color="#dc322f", alpha=0.85, linewidth=0, interpolate=True)
    ax.fill_between(x, 0, a, where=a < 0, color="#268bd2", alpha=0.85, linewidth=0, interpolate=True)
    ax.axhline(0, color="#666", lw=0.5)
    if 0 < fence < len(a):
        ax.axvline(fence, color="#859900", lw=1.1, ls="--", label="code fence")
        ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("attempt token index"); ax.set_ylabel("A_t")
    ax.margins(x=0)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="scratch_tadv/teacher_eval.json")
    ap.add_argument("--rollouts", default="reports/iteration-05/data/qwen3_rollouts.json")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="scratch_tadv/advantage_view.html")
    ap.add_argument("--ids", default="", help="comma loj ids to include (default all)")
    args = ap.parse_args()

    ev = json.load(open(args.eval))
    roll = {(r["id"], r["sample"]): r for r in json.load(open(args.rollouts))["results"]}
    tok = AutoTokenizer.from_pretrained(args.model)
    want = {int(i) for i in args.ids.split(",")} if args.ids else None

    parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font:14px/1.5 -apple-system,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}",
        ".attempt{white-space:pre-wrap;word-break:break-word;font:12px/1.55 ui-monospace,Menlo,monospace;",
        "  border:1px solid #ddd;border-radius:6px;padding:12px;background:#fff}",
        "pre{white-space:pre-wrap;background:#f6f8fa;padding:10px;border-radius:6px;font-size:12px}",
        "h2{border-top:2px solid #eee;padding-top:18px;margin-top:28px}",
        "details>summary{cursor:pointer;color:#268bd2}.legend span{padding:1px 6px;border-radius:3px}",
        "</style></head><body>",
        f"<h1>SDPO advantage A_t over the attempt — {html.escape(ev.get('critic_set','?'))} critique</h1>",
        "<p class='legend'>teacher = <code>x + critique</code>, student = <code>x</code>; "
        "A_t = log q(yₜ|x,critique,y&lt;t) − log π(yₜ|x,y&lt;t). "
        "<span style='background:rgba(220,50,47,.7)'>red = A_t&gt;0 (keep)</span> &nbsp; "
        "<span style='background:rgba(38,139,210,.7)'>blue = A_t&lt;0 (change / flaw)</span></p>",
    ]

    for r in ev["results"]:
        if want and r["id"] not in want:
            continue
        pid, s, lang = r["id"], r["sample"], r["language"]
        rk = roll.get((pid, s))
        if not rk:
            continue
        attempt = rk["completion"]
        critique = r["critique"]
        adv = r["advantage"]
        a_t, slp, tlp = adv["A_t"], adv["student_lp"], adv["teacher_lp"]
        fence = adv.get("code_fence_tok", len(a_t))
        problem = (CPP_PROMPT_BY_ID if lang == "cpp" else PROMPT_BY_ID)[pid]

        y_ids = tok.encode(attempt, add_special_tokens=False)
        n = min(len(y_ids), len(a_t))
        srt = sorted(abs(v) for v in a_t[:n])
        vmax = srt[int(0.97 * (n - 1))] if n else 1.0
        vmax = max(vmax, 1e-6)

        spans = []
        for i in range(n):
            piece = html.escape(tok.decode([y_ids[i]]))
            ttl = f"A_t={a_t[i]:.2f} t={tlp[i]:.2f} s={slp[i]:.2f}"
            spans.append(f"<span title='{ttl}' style='background:{color(a_t[i], vmax)}'>{piece}</span>")

        parts += [
            f"<h2>{r['difficulty']} loj-{pid} s{s} — base {r['base_verdict']} "
            f"(reward {r['base_reward']})</h2>",
            f"<p><b>solve-rate (teacher | x+critique, K):</b> {r['solve_rate']} &nbsp;·&nbsp; "
            f"<b>A_t</b> mean|·|={adv['mean_abs']} frac_neg={adv['frac_neg']} "
            f"code/reas={adv['code_region_abs']}/{adv['reasoning_abs']} &nbsp;·&nbsp; "
            f"attempt {n} tok, code fence @ {fence} &nbsp;·&nbsp; A_t color-scale ±{vmax:.2f}</p>",
            f"<img src='data:image/png;base64,{at_plot_b64(a_t[:n], fence)}' style='width:100%'>",
            "<h4>Critic feedback (what the teacher is conditioned on)</h4>",
            f"<pre>{html.escape(critique)}</pre>",
            "<details><summary>Problem statement</summary>"
            f"<pre>{html.escape(problem)}</pre></details>",
            "<h4>Attempt — shaded by A_t</h4>",
            f"<div class='attempt'>{''.join(spans)}</div>",
        ]

    parts.append("</body></html>")
    out = Path(args.out)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
