"""iteration-08: build a tightened frontier_band_v2.json from the binary-solve-rate probe.

Reads the probe json (per-problem n_ac at n samples, temp 1.0), computes solve rate p=n_ac/n,
and selects the BINARY-FRONTIER (p in [lo,hi]) — the problems whose 8 training rollouts genuinely
split under the binary reward, minimizing flat groups (P(flat@8)=p^8+(1-p)^8). Excludes saturated
(p>hi, all-AC → flat), hopeless (p=0), and weak (0<p<lo, mostly flat). Writes the band + buckets +
predicted flat_group, matching frontier_band.json's schema so --frontier-band reads ["frontier_band"].

Usage: python src/build_frontier_v2.py [probe.json] [lo] [hi]
"""
import json
import sys


def flatprob(p, k=8):
    return p ** k + (1 - p) ** k


def main():
    probe = sys.argv[1] if len(sys.argv) > 1 else "sdpo_passk_probe_v2.json"
    lo = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    hi = float(sys.argv[3]) if len(sys.argv) > 3 else 0.75
    res = json.load(open(probe))["results"]

    buckets = {"frontier_band": [], "saturated": [], "weak_frontier": [], "hopeless": []}
    per = {}
    for r in res:
        n = max(1, r["n"]); p = r["n_ac"] / n
        per[str(r["id"])] = {"p": round(p, 3), "n_ac": r["n_ac"], "n": r["n"],
                             "difficulty": r["difficulty"]}
        if p == 0:
            buckets["hopeless"].append(r["id"])
        elif p > hi:
            buckets["saturated"].append(r["id"])
        elif p < lo:
            buckets["weak_frontier"].append(r["id"])
        else:
            buckets["frontier_band"].append(r["id"])
    for k in buckets:
        buckets[k] = sorted(buckets[k])

    band = buckets["frontier_band"]
    pflat_band = (sum(flatprob(per[str(i)]["p"]) for i in band) / len(band)) if band else 0.0
    pflat_all = sum(flatprob(per[str(r["id"])]["p"]) for r in res) / len(res)

    out = {
        "frontier_band": band,
        "saturated": buckets["saturated"],
        "weak_frontier": buckets["weak_frontier"],
        "hopeless": buckets["hopeless"],
        "probe": {"n": res[0]["n"], "temperature": 1.0, "lo": lo, "hi": hi,
                  "note": "iter-08 binary solve-rate probe (temp 1.0); band = p in [lo,hi]"},
        "predicted_flat_group": {"band_v2": round(pflat_band, 3), "full_pool": round(pflat_all, 3)},
        "per_problem": per,
    }
    json.dump(out, open("data/frontier_band_v2.json", "w"), indent=2)

    print(f"probe: {len(res)} problems, n={res[0]['n']}, band cutoff p in [{lo},{hi}]")
    for k, v in buckets.items():
        print(f"  {k:16s}: {len(v):>2}  {v}")
    print(f"\nfrontier_band_v2: {len(band)} problems")
    print(f"predicted mean P(flat@8): band_v2={pflat_band:.2f}  (vs iter-07 band ~0.71 theoretical / 0.44 observed)")
    print("wrote data/frontier_band_v2.json")


if __name__ == "__main__":
    main()
