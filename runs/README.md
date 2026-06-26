# runs/ — per-iteration local outputs (gitignored)

Raw outputs from each iteration live here so runs never overwrite each other:

```
runs/iteration-02/
  sdpo_out/                 # trained LoRA adapter
  *.log                     # training / eval / serve logs
  sdpo_passk_*.json         # raw pass@k results
  sdpo_eval_*.json          # raw held-out pass@1
  results_gsm8k_*.json      # raw GSM8K
```

**Convention:** `cd runs/iteration-NN/` before running training/eval so CWD-relative
outputs land here (the scripts write to the current directory).

Everything in `runs/` is gitignored. **Curated, committed results** (small CSVs, eval
summaries, figures, the report) go to [`../reports/iteration-NN/`](../reports/) instead.
Trained adapters are preserved durably on the Modal `sdpo-outputs` volume under a
per-iteration path (see the iteration's `PROVENANCE.md`).
