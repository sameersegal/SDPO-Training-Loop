#!/usr/bin/env python3
"""Slide-ready broken-x-axis view of the per-token SDPO advantage A_t for ONE case.

A full attempt (e.g. loj-2129 is 11,115 tokens, ~96% of it one `<think>` block) is far
too long to read as a single A_t strip — the few informative patches of blue (A_t<0,
"change this") get crushed to invisible slivers. This crops the attempt to a handful of
segments and lays them side-by-side with axis-break marks, so a slide can show the few
blue patches in the reasoning AND the code tail at a legible scale.

A_t is read straight from a teacher_eval.json (already computed — no GPU here). Segments
are auto-picked (top blue-density windows in the reasoning region + the code section) or
given explicitly with --segments.

  S=../../src
  PYTHONPATH=$S python $S/plot_advantage_broken.py            # loj-2129, auto segments
  PYTHONPATH=$S python $S/plot_advantage_broken.py --id 2129 \
      --segments 90-320,520-640,10760-11115 --out fig.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from transformers import AutoTokenizer

from sdpo_ojbench import PROMPT_BY_ID, CPP_PROMPT_BY_ID  # noqa: F401 (kept for parity/use)

RED, BLUE, FENCE_C = "#dc322f", "#268bd2", "#859900"


def parse_segments(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out


def auto_segments(a, fence, n_patches, patch_w, code_pad, min_gap):
    """Pick the densest-blue windows in the reasoning region, plus the code section.

    Returns a left-to-right-sorted list of (start, end) token spans. Reasoning patches
    are greedy by negative-mass and kept at least `min_gap` tokens apart, so the chosen
    patches are genuinely discontinuous (the whole point of the broken axis). The code
    span is fence-code_pad..end."""
    n = len(a)
    fence = fence if (fence and 0 < fence < n) else n
    W = patch_w
    # negative mass in each W-window over the reasoning region only
    cand = []
    i = 0
    while i + W <= fence:
        seg = a[i:i + W]
        cand.append((i, float(seg[seg < 0].sum())))  # more negative = more blue
        i += W // 2  # 50% overlap so a patch isn't split across stride boundaries
    cand.sort(key=lambda x: x[1])
    chosen = []
    for start, _ in cand:
        s, e = start, start + W
        if any(s < ue + min_gap and e + min_gap > us for us, ue in chosen):  # too close
            continue
        chosen.append((s, e))
        if len(chosen) >= n_patches:
            break
    chosen.sort()
    code_start = max(0, fence - code_pad)
    chosen.append((code_start, n))
    return chosen


def draw_panel(ax, a, s, e, fence, vmax, is_code):
    x = np.arange(s, e)
    seg = a[s:e]
    ax.fill_between(x, 0, seg, where=seg >= 0, color=RED, alpha=0.85,
                    linewidth=0, interpolate=True)
    ax.fill_between(x, 0, seg, where=seg < 0, color=BLUE, alpha=0.85,
                    linewidth=0, interpolate=True)
    ax.axhline(0, color="#666", lw=0.5)
    if s <= fence < e:
        ax.axvline(fence, color=FENCE_C, lw=1.3, ls="--")
        ax.text(fence, -vmax * 0.93, " code →", color=FENCE_C, fontsize=8,
                va="bottom", ha="left")
    ax.set_xlim(s, e)
    ax.set_ylim(-vmax, vmax)
    # boundary ticks of adjacent panels land at the same screen x; left-align the
    # start label and right-align the end label so they read as two separate numbers.
    ax.set_xticks([s, e - 1])
    ax.set_xticklabels([f"{s:,}", f"{e - 1:,}"], fontsize=7)
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[1].set_ha("right")
    ax.tick_params(axis="y", labelsize=7)
    label = "code section" if is_code else f"reasoning · tok {s:,}–{e:,}"
    ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=8,
            va="top", ha="left", color="#333",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", lw=0.5, alpha=0.85))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="reports/iteration-05/data/teacher_eval.json")
    ap.add_argument("--rollouts", default="reports/iteration-05/data/qwen3_rollouts.json")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--id", type=int, default=2129)
    ap.add_argument("--sample", type=int, default=None, help="default: first matching id")
    ap.add_argument("--segments", default="", help="comma spans a-b,a-b (overrides auto)")
    ap.add_argument("--n-patches", type=int, default=2, help="auto: # reasoning blue patches")
    ap.add_argument("--patch-w", type=int, default=200, help="auto: tokens per reasoning patch")
    ap.add_argument("--min-gap", type=int, default=150,
                    help="auto: min token gap between reasoning patches (forces breaks)")
    ap.add_argument("--code-pad", type=int, default=40, help="tokens shown before the code fence")
    ap.add_argument("--out", default="reports/iteration-05/figures/advantage_2129_broken.png")
    args = ap.parse_args()

    ev = json.load(open(args.eval))
    rows = [r for r in ev["results"] if r["id"] == args.id
            and (args.sample is None or r["sample"] == args.sample)]
    if not rows:
        raise SystemExit(f"id {args.id} (sample {args.sample}) not in {args.eval}")
    r = rows[0]
    adv = r["advantage"]
    a = np.asarray(adv["A_t"], dtype=float)
    fence = adv.get("code_fence_tok") or len(a)

    # color/y scale: clip to the 97th pct of |A_t| so a couple of huge spikes don't
    # flatten everything (same convention as plot_advantage_text.py).
    srt = np.sort(np.abs(a))
    vmax = max(float(srt[int(0.97 * (len(a) - 1))]), 1e-6)

    segs = (parse_segments(args.segments) if args.segments
            else auto_segments(a, fence, args.n_patches, args.patch_w,
                               args.code_pad, args.min_gap))
    segs = [(max(0, s), min(len(a), e)) for s, e in segs]
    print(f"loj-{args.id} s{r['sample']} {r['difficulty']} — {len(a)} tok, fence @ {fence}")
    print(f"segments: {segs}  (y/​color scale ±{vmax:.2f})")

    widths = [e - s for s, e in segs]
    fig = plt.figure(figsize=(12.5, 3.0))
    gs = GridSpec(1, len(segs), width_ratios=widths, wspace=0.06)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(segs))]
    for i, (ax, (s, e)) in enumerate(zip(axes, segs)):
        draw_panel(ax, a, s, e, fence, vmax, is_code=(s <= fence < e and e - fence < 600))
        if i == 0:
            ax.set_ylabel("A_t  (nats)", fontsize=9)
        else:
            ax.tick_params(labelleft=False)
            ax.spines["left"].set_visible(False)
        if i != len(segs) - 1:
            ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # diagonal break marks between adjacent panels
        d = 0.012
        kw = dict(transform=ax.transAxes, color="#999", clip_on=False, lw=1)
        if i != len(segs) - 1:
            ax.plot([1 - d, 1 + d], [-d, d], **kw)
            ax.plot([1 - d, 1 + d], [1 - d, 1 + d], **kw)
        if i != 0:
            ax.plot([-d, d], [-d, d], **kw)
            ax.plot([-d, d], [1 - d, 1 + d], **kw)

    fig.suptitle(
        f"loj-{args.id} ({r['difficulty']}, base WA) — per-token advantage  "
        f"A_t = log q(y|x,critique) − log π(y|x)",
        fontsize=11, y=0.99)
    fig.text(0.5, 0.91,
             f"red = keep (A_t>0) · blue = change/flaw (A_t<0) · x-axis broken: "
             f"{len(a):,} tok total, code is only the last {len(a)-fence}",
             ha="center", fontsize=8.5, color="#555")
    fig.text(0.5, 0.01, "attempt token index (axis broken between panels)",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 0.82])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
