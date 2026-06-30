"""Cross-iteration comparison graphs (iter-05 vs 06 vs 07) for the iteration-07 report.

Panels:
  (1) train==eval pass@8: base vs ckpt for each iteration (the defining metric).
  (2) completion mean-length trajectory (training dynamics).
  (3) flat_group_fraction trajectory (the mechanism — frontier band should lower it).

pass@8 for iter-06/07 is recomputed from the committed sdpo_passk_*_iterNNprobe jsons on the
12-problem probe; iter-05's train==eval (0.83→0.50) is from its REPORT (same probe). Run after
the iter-07 eval. Writes reports/iteration-07/figures/iter05_06_07_comparison.png.
"""
import csv
import glob
import json
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROBE = [2132, 2297, 2361, 2415, 2420, 2423, 2594, 2595, 2602, 3387, 3896, 4001]


def pass_at_k(n, c, k):
    k = min(k, n)
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def passk_from_json(path, k=8):
    if not path:
        return None
    res = json.load(open(path))["results"]
    rs = [r for r in res if r["id"] in PROBE]
    return sum(pass_at_k(r["n"], r["n_ac"], k) for r in rs) / len(rs) if rs else None


def find(globpat, exclude=None):
    fs = [f for f in glob.glob(globpat) if not (exclude and exclude in f)]
    return fs[0] if fs else None


def traj(csvpath, key):
    out = []
    try:
        for r in csv.DictReader(open(csvpath)):
            if not r.get("train/global_step"):
                continue
            try:
                out.append((int(float(r["train/global_step"])), float(r[key])))
            except Exception:
                pass
    except FileNotFoundError:
        pass
    return out


def main():
    D = "reports/iteration-06/data"
    D7 = "reports/iteration-07/data"
    # pass@8 (base, ckpt) per iteration
    p = {
        "iter-05": (0.83, 0.50),  # from reports/iteration-05 (same 12-probe)
        "iter-06": (passk_from_json(f"{D}/sdpo_passk_base_iter06probe.json"),
                    passk_from_json(find(f"{D}/sdpo_passk_*iter06probe.json", exclude="base"))),
        "iter-07": (passk_from_json(find(f"{D7}/sdpo_passk_base_iter07probe.json")
                                    or "reports/iteration-07/data/sdpo_passk_base_iter07probe.json"),
                    passk_from_json(find(f"{D7}/sdpo_passk_*iter07probe.json", exclude="base"))),
    }
    print("pass@8 (base -> ckpt):")
    for k, (b, c) in p.items():
        print(f"  {k}: {b} -> {c}" + (f"  (Δ{c-b:+.3f})" if b and c else ""))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("iteration-05 → 06 → 07: collapse → attenuate → frontier-band (Qwen3-8B SDPO)",
                 fontweight="bold")

    # Panel 1: pass@8 base vs ckpt
    a = ax[0]
    labels = list(p.keys())
    x = range(len(labels))
    bs = [p[k][0] for k in labels]
    cs = [p[k][1] for k in labels]
    w = 0.38
    a.bar([i - w / 2 for i in x], bs, w, label="base", color="slategray")
    a.bar([i + w / 2 for i in x], cs, w, label="ckpt", color="seagreen")
    for i, k in enumerate(labels):
        b, c = p[k]
        if b:
            a.text(i - w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
        if c:
            a.text(i + w / 2, c + 0.01, f"{c:.2f}", ha="center", fontsize=9)
            if b:
                a.annotate(f"Δ{c-b:+.2f}", xy=(i, max(b, c) + 0.06), ha="center", fontsize=9,
                           color="crimson" if c < b else "green")
    a.set_ylim(0, 1.05); a.set_xticks(list(x)); a.set_xticklabels(labels)
    a.set_ylabel("train==eval pass@8"); a.set_title("pass@8: base vs ckpt (the verdict)")
    a.legend(fontsize=9); a.grid(axis="y", alpha=0.3)

    # Panel 2: length trajectory
    a = ax[1]
    for name, f, col in [("iter-05", f"{D}/iter05_train_history_8281dbd7.csv", "crimson"),
                         ("iter-06", f"{D}/iter06_train_history_ds1kqb6v.csv", "darkorange"),
                         ("iter-07", f"{D7}/iter07_train_history.csv", "seagreen")]:
        t = traj(f, "train/completions/mean_length")
        if t:
            a.plot([s for s, _ in t], [v for _, v in t], "o-", color=col, label=name)
    a.axhspan(0, 8000, color="crimson", alpha=0.05)
    a.set_title("completion mean length / step"); a.set_xlabel("step"); a.set_ylabel("tokens")
    a.legend(fontsize=9); a.grid(alpha=0.3)

    # Panel 3: flat_group trajectory (mechanism)
    a = ax[2]
    for name, f, col in [("iter-06", f"{D}/iter06_train_history_ds1kqb6v.csv", "darkorange"),
                         ("iter-07 (frontier)", f"{D7}/iter07_train_history.csv", "seagreen")]:
        t = traj(f, "train/self_distillation/flat_group_fraction")
        if t:
            a.plot([s for s, _ in t], [v for _, v in t], "s-", color=col, label=name)
    a.set_title("flat_group_fraction (lower = more policy-gradient signal)")
    a.set_xlabel("step"); a.set_ylabel("flat_group_fraction"); a.set_ylim(-0.05, 1.05)
    a.legend(fontsize=9); a.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = "reports/iteration-07/figures/iter05_06_07_comparison.png"
    import os
    os.makedirs("reports/iteration-07/figures", exist_ok=True)
    plt.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
