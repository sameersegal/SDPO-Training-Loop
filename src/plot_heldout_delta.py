"""Slide figure: held-out pass@k, base vs an SDPO adapter, by difficulty + overall.

Reads two sdpo_passk_*.json (base + adapter) and draws pass@1/2/4/8 curves so the
base-vs-adapter delta (the iteration deliverable) is visible at a glance.

    python src/plot_heldout_delta.py \
        --base reports/iteration-05/data/sdpo_passk_ckpt20eval_base_easymed.json \
        --adapter reports/iteration-05/data/sdpo_passk_ckpt20_easymed.json \
        --adapter-label "SDPO+critic ckpt-20" \
        --title "Held-out pass@k — Qwen3-8B, 20 steps, Python (easy+medium)" \
        --out reports/iteration-05/figures/heldout_passk_base_vs_ckpt20.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


KS = [1, 2, 4, 8]


def _curve(summary, lang, bucket):
    d = summary["by_language"][lang].get(bucket, {})
    return [d.get(f"pass@{k}") for k in KS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--adapter-label", default="SDPO adapter")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--title", default="Held-out pass@k — base vs adapter")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = json.load(open(args.base))["summary"]
    adpt = json.load(open(args.adapter))["summary"]
    buckets = [b for b in ("easy", "medium", "overall")
               if b in base["by_language"][args.lang]]

    fig, axes = plt.subplots(1, len(buckets), figsize=(4.6 * len(buckets), 4.4), sharey=True)
    if len(buckets) == 1:
        axes = [axes]
    x = list(range(len(KS)))

    for ax, bucket in zip(axes, buckets):
        b = _curve(base, args.lang, bucket)
        a = _curve(adpt, args.lang, bucket)
        n = base["by_language"][args.lang][bucket].get("n_problems", "?")
        ax.plot(x, b, "o-", color="#555555", lw=2.4, ms=8, label="base")
        ax.plot(x, a, "s--", color="#1f77b4", lw=2.4, ms=8, label=args.adapter_label)
        # value labels at the endpoints
        for xi, (bv, av) in enumerate(zip(b, a)):
            if xi in (0, len(KS) - 1):
                ax.annotate(f"{bv:.2f}", (xi, bv), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=9, color="#555555")
                ax.annotate(f"{av:.2f}", (xi, av), textcoords="offset points",
                            xytext=(0, -16), ha="center", fontsize=9, color="#1f77b4")
        ax.set_title(f"{bucket}  (n={n})", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"pass@{k}" for k in KS])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("k (samples)")
    axes[0].set_ylabel("pass@k  (private-test AC rate)")
    axes[0].legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.suptitle(args.title, fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
