#!/usr/bin/env python3
"""Aggregate the per-run multi-seed eval rows into one table with error bars.

Reads every <out>/worm_ablations/eval_rows/*.csv (one row each, written by the
eval_multiseed.sbatch array), concatenates them into eval_metrics_multiseed.csv,
then groups by base config (stripping __seed<N>) and reports mean +/- std of
test Spearman / Pearson / loss across seeds.

seed1337 is NOT present in eval_rows (the manifest only lists seeds 1/2/3), so
no exclusion is needed here -- error bars are over the full-valid seeds only.

Usage: python experiments/worm_ablations/aggregate_multiseed_eval.py
"""
import csv, glob, os, re
from collections import defaultdict

from experiments.common.paths import out_dir
ROWDIR = str(out_dir("worm_ablations", "eval_rows"))
COMBINED = str(out_dir("worm_ablations") / "eval_metrics_multiseed.csv")
TABLE = str(out_dir("worm_ablations") / "eval_multiseed_table.md")

ANCHOR = "magic_mi_knn3_scatter"


def mean_std(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5


def main():
    files = sorted(glob.glob(os.path.join(ROWDIR, "*.csv")))
    if not files:
        print(f"No per-run eval csvs under {ROWDIR}/ -- has the array job finished?")
        return

    rows = []
    for f in files:
        with open(f) as fh:
            r = list(csv.DictReader(fh))
        if r:
            rows.append(r[0])

    # write combined raw csv
    fields = list(rows[0].keys())
    with open(COMBINED, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["config"]):
            w.writerow(r)

    # group by base config (strip __seed<N>)
    groups = defaultdict(list)
    for r in rows:
        base = re.sub(r"__seed\d+$", "", r["config"])
        groups[base].append(r)

    def agg(base, key):
        vals = []
        for r in groups[base]:
            try:
                vals.append(float(r[key]))
            except (ValueError, KeyError, TypeError):
                pass
        return mean_std(vals)

    a_sp_m, a_sp_s = agg(ANCHOR, "test_spearman")

    lines = ["# Multi-seed eval (held-out test set, mean +/- std across seeds)", ""]
    lines.append(f"Anchor (`{ANCHOR}`) test Spearman: **{a_sp_m:.4f} +/- {a_sp_s:.4f}** "
                 f"(n={len(groups.get(ANCHOR, []))} seeds).")
    lines.append("`~noise` = |Δ Spearman| < (anchor_std + variant_std). Higher Spearman = better.")
    lines.append("")
    lines.append("| config | n | test Spearman (mean±std) | test Pearson (mean±std) | test loss (mean±std) | Δ Spearman | verdict |")
    lines.append("|---|---|---|---|---|---|---|")

    for base in sorted(groups, key=lambda b: -agg(b, "test_spearman")[0]):
        n = len(groups[base])
        sp_m, sp_s = agg(base, "test_spearman")
        pe_m, pe_s = agg(base, "test_pearson")
        lo_m, lo_s = agg(base, "test_loss")
        d = sp_m - a_sp_m
        if base == ANCHOR:
            verdict = "⭐ anchor"
        elif abs(d) < (a_sp_s + sp_s):
            verdict = "~ within noise"
        elif d > 0:
            verdict = "BETTER than anchor"
        else:
            verdict = "worse than anchor"
        lines.append(f"| {base} | {n} | {sp_m:.4f} ± {sp_s:.4f} | {pe_m:.4f} ± {pe_s:.4f} | "
                     f"{lo_m:.4f} ± {lo_s:.4f} | {d:+.4f} | {verdict} |")

    md = "\n".join(lines)
    with open(TABLE, "w") as fh:
        fh.write(md + "\n")
    print(md)
    print(f"\nwrote {COMBINED}\nwrote {TABLE}")
    print(f"\n[aggregate] {len(rows)} per-run rows across {len(groups)} configs")


if __name__ == "__main__":
    main()
