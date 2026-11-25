#!/usr/bin/env python3
"""Aggregate the multi-seed expression ablation into a table with error bars.

Motivation: the single-seed R2 table picked the global minimum of a *very*
jagged valid-loss curve (valid swings by 10+ between adjacent evals), so a
0.02-0.04 spread between encoders was being read as a ranking when it was
within run-to-run noise. This script:

  1. Reads every <runs>/expression/<config>__seed<S>/model/loss_dict.json
  2. Reports BOTH the raw min and a smoothed min (min of a rolling-mean over
     `WINDOW` evals) so the metric isn't a single lucky eval.
  3. Aggregates across seeds -> mean +/- std of the smoothed best valid loss.
  4. Flags any non-anchor variant whose gap to the anchor is within the
     combined std ("~ within noise") -- i.e. NOT a real difference.

Usage: python experiments/worm_ablations/make_multiseed_table.py
"""
import json, glob, os, re
from collections import defaultdict

from experiments.common.paths import out_dir
from experiments.worm_ablations.common import abl_runs
EXPDIR = str(abl_runs() / "expression")
STCDIR = str(abl_runs() / "structure")
WINDOW = 3  # rolling-mean window for the smoothed minimum

# config -> (axis, human label, is_anchor). stc_500 == the expression anchor
# (same scatter/magic/mi_knn3/500 config), so it appears once as the anchor and
# is reused as the structure-axis baseline rather than run separately.
SPEC = {
    "magic_mi_knn3_scatter":     ("anchor",       "MAGIC + MI-kNN3 + scatter (= stc_500)", True),
    "raw_log1p_mi_knn3_scatter": ("denoiser",     "raw log1p (no MAGIC)", False),
    "knn_smooth_mi_knn3_scatter":("denoiser",     "kNN smoothing", False),
    "magic_mi_knn5_scatter":     ("graph metric", "MI-kNN5", False),
    "magic_mi_knn10_scatter":    ("graph metric", "MI-kNN10", False),
    "magic_mi_binned10_scatter": ("graph metric", "MI-binned10", False),
    "magic_pearson_scatter":     ("graph metric", "Pearson", False),
    "magic_spearman_scatter":    ("graph metric", "Spearman", False),
    "magic_mi_knn3_gnn":         ("encoder",      "GNN (GCNConv, +graph)", False),
    "magic_mi_knn3_mlp_node":    ("encoder",      "MLP on node feats (fair, no graph)", False),
    "magic_mi_knn3_mlp_scalar":  ("encoder",      "MLP on per-gene scalar (input-starved)", False),
    "stc_750":                   ("structure window", "750 bp", False),
    "stc_1000":                  ("structure window", "1000 bp", False),
}
AXIS_ORDER = ["anchor", "denoiser", "graph metric", "encoder", "structure window"]


def smoothed_min(valid, window=WINDOW):
    if len(valid) < window:
        return min(valid)
    roll = [sum(valid[i:i + window]) / window for i in range(len(valid) - window + 1)]
    return min(roll)


def mean_std(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5


def collect():
    # config -> list of (seed, raw_min, smooth_min, last_step)
    runs = defaultdict(list)
    seeded_dirs = (glob.glob(os.path.join(EXPDIR, "*__seed*"))
                   + glob.glob(os.path.join(STCDIR, "*__seed*")))
    for d in sorted(seeded_dirs):
        base = os.path.basename(d)
        m = re.match(r"(.+)__seed(\d+)$", base)
        if not m:
            continue
        config, seed = m.group(1), int(m.group(2))
        # seed1337 is deprecated from results: it was logged under the old
        # partial-valid criterion (noisy ~1600-row subset -> jagged curve,
        # different eval cadence, higher floor) and is NOT comparable to the
        # full-valid seeds 1/2/3. Exclude it from the error bars.
        if seed == 1337:
            continue
        f = os.path.join(d, "model", "loss_dict.json")
        if not os.path.exists(f):
            continue
        ld = json.load(open(f))
        v = ld.get("valid_loss", [])
        if not v:
            continue
        runs[config].append((seed, min(v), smoothed_min(v), ld["step"][-1]))
    return runs


def main():
    runs = collect()
    if not runs:
        print(f"No multi-seed runs found under {EXPDIR}/*__seed*/")
        return

    # anchor stats
    anchor_vals = [s for (_, _, s, _) in runs.get("magic_mi_knn3_scatter", [])]
    a_mean, a_std = mean_std(anchor_vals)

    lines = ["# Multi-seed expression ablation (smoothed best valid loss)", ""]
    lines.append(f"Smoothed best valid = min of rolling-mean(window={WINDOW}) over the valid curve.")
    lines.append(f"Anchor: **{a_mean:.4f} ± {a_std:.4f}** (n={len(anchor_vals)} seeds).")
    lines.append("Δ = variant_mean − anchor_mean. `~noise` = |Δ| < (anchor_std + variant_std).")
    lines.append("")

    for axis in AXIS_ORDER:
        configs = [c for c in SPEC if SPEC[c][0] == axis and c in runs]
        if not configs:
            continue
        lines.append(f"## {axis}")
        lines.append("")
        lines.append("| variant | smoothed valid (mean±std) | n seeds | raw-min mean | Δ vs anchor | verdict |")
        lines.append("|---|---|---|---|---|---|")
        for c in configs:
            recs = runs[c]
            smooth = [s for (_, _, s, _) in recs]
            raw = [r for (_, r, _, _) in recs]
            m, sd = mean_std(smooth)
            rm, _ = mean_std(raw)
            delta = m - a_mean
            anchor = SPEC[c][2]
            if anchor:
                verdict = "⭐ anchor"
            elif abs(delta) < (a_std + sd):
                verdict = "~ within noise"
            elif delta < 0:
                verdict = "BETTER than anchor"
            else:
                verdict = "worse than anchor"
            lines.append(
                f"| {SPEC[c][1]} | {m:.4f} ± {sd:.4f} | {len(recs)} | "
                f"{rm:.4f} | {delta:+.4f} | {verdict} |")
        lines.append("")

    md = "\n".join(lines)
    out = str(out_dir("worm_ablations") / "multiseed_table.md")
    open(out, "w").write(md + "\n")
    print(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
