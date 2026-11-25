#!/usr/bin/env python3
"""Plots + summary tables for the multi-seed worm eval.

Reads <out>/worm_ablations/eval_metrics_multiseed.csv (per-run rows, seeds 1/2/3 --
seed1337 excluded), groups by base config, and shows that NO ablation axis
separates from the anchor beyond seed noise.

Outputs (under <out>/worm_ablations/plots/):
  - test_spearman_by_axis.png  : one panel per axis, bars=configs, err=±std,
                                 dashed line + shaded band = anchor mean ± std
  - test_pearson_by_axis.png   : same for Pearson
  - summary_by_axis.csv / .md  : mean±std per config with Δ-vs-anchor + verdict

Run under the mioflow conda env (needs matplotlib/numpy/pandas).
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common.paths import out_dir
CSV = str(out_dir("worm_ablations") / "eval_metrics_multiseed.csv")
PLOTDIR = str(out_dir("worm_ablations", "plots"))
os.makedirs(PLOTDIR, exist_ok=True)

ANCHOR = "magic_mi_knn3_scatter"

# config -> (axis, short label). Anchor appears in its own panel AND is the
# baseline band on every other panel.
SPEC = {
    "magic_mi_knn3_scatter":      ("anchor",          "MAGIC+MIkNN3\nscatter (anchor)"),
    "raw_log1p_mi_knn3_scatter":  ("denoiser",        "raw log1p"),
    "knn_smooth_mi_knn3_scatter": ("denoiser",        "kNN smooth"),
    "magic_mi_knn5_scatter":      ("graph metric",    "MI-kNN5"),
    "magic_mi_knn10_scatter":     ("graph metric",    "MI-kNN10"),
    "magic_mi_binned10_scatter":  ("graph metric",    "MI-binned10"),
    "magic_pearson_scatter":      ("graph metric",    "Pearson"),
    "magic_spearman_scatter":     ("graph metric",    "Spearman"),
    "magic_mi_knn3_gnn":          ("encoder",         "GNN"),
    "magic_mi_knn3_mlp_node":     ("encoder",         "MLP-node"),
    "magic_mi_knn3_mlp_scalar":   ("encoder",         "MLP-scalar"),
    "stc_750":                    ("structure window", "750 bp"),
    "stc_1000":                   ("structure window", "1000 bp"),
}
AXES = ["denoiser", "graph metric", "encoder", "structure window"]


def load():
    df = pd.read_csv(CSV)
    df["base"] = df["config"].str.replace(r"__seed\d+$", "", regex=True)
    return df


def agg(df, metric):
    g = df.groupby("base")[metric].agg(["mean", "std", "count"]).fillna(0.0)
    return g


def plot_metric(df, metric, title, fname):
    g = agg(df, metric)
    a_m, a_s = g.loc[ANCHOR, "mean"], g.loc[ANCHOR, "std"]

    fig, axarr = plt.subplots(1, len(AXES), figsize=(4.2 * len(AXES), 4.6), sharey=True)
    for ax, axis in zip(axarr, AXES):
        cfgs = [c for c in SPEC if SPEC[c][0] == axis and c in g.index]
        # always show the anchor bar first as the in-axis baseline
        bars = [ANCHOR] + cfgs
        labels = [SPEC[ANCHOR][1]] + [SPEC[c][1] for c in cfgs]
        means = [g.loc[c, "mean"] for c in bars]
        stds = [g.loc[c, "std"] for c in bars]
        x = np.arange(len(bars))
        colors = ["#444444"] + ["#4C77B0"] * len(cfgs)
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
        # anchor band
        ax.axhspan(a_m - a_s, a_m + a_s, color="#444444", alpha=0.12, zorder=0)
        ax.axhline(a_m, color="#444444", ls="--", lw=1, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(axis, fontsize=11)
        ax.grid(axis="y", ls=":", alpha=0.4)
    axarr[0].set_ylabel(title)
    # full 0..1 scale: these metrics are bounded, and an honest y-axis makes the
    # tiny cross-config differences read as tiny (a zoomed axis exaggerates them).
    axarr[0].set_ylim(0.0, 1.0)
    fig.suptitle(f"{title} across worm ablation axes (n=seeds shown; band = anchor ± std)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(PLOTDIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def write_tables(df):
    rows = []
    for metric in ["test_spearman", "test_pearson", "test_loss"]:
        g = agg(df, metric)
        a_m = g.loc[ANCHOR, "mean"]; a_s = g.loc[ANCHOR, "std"]
        for base in g.index:
            if metric == "test_spearman":
                rows.append({"config": base, "axis": SPEC.get(base, ("?",))[0],
                             "n_seeds": int(g.loc[base, "count"])})
    summ = {r["config"]: r for r in rows}
    for metric in ["test_spearman", "test_pearson", "test_loss"]:
        g = agg(df, metric)
        a_m = g.loc[ANCHOR, "mean"]; a_s = g.loc[ANCHOR, "std"]
        for base in g.index:
            m, s = g.loc[base, "mean"], g.loc[base, "std"]
            summ[base][f"{metric}_mean"] = round(m, 4)
            summ[base][f"{metric}_std"] = round(s, 4)
            if metric == "test_spearman":
                d = m - a_m
                summ[base]["d_spearman"] = round(d, 4)
                if base == ANCHOR:
                    summ[base]["verdict"] = "anchor"
                elif abs(d) < (a_s + s):
                    summ[base]["verdict"] = "within noise"
                else:
                    summ[base]["verdict"] = "BETTER" if d > 0 else "worse"
    order = {a: i for i, a in enumerate(["anchor"] + AXES)}
    out = pd.DataFrame(summ.values())
    out["_o"] = out["axis"].map(lambda a: order.get(a, 99))
    out = out.sort_values(["_o", "test_spearman_mean"], ascending=[True, False]).drop(columns="_o")
    cols = ["config", "axis", "n_seeds", "test_spearman_mean", "test_spearman_std",
            "test_pearson_mean", "test_pearson_std", "test_loss_mean", "test_loss_std",
            "d_spearman", "verdict"]
    out = out[cols]
    csv_out = os.path.join(PLOTDIR, "summary_by_axis.csv")
    out.to_csv(csv_out, index=False)
    md_out = os.path.join(PLOTDIR, "summary_by_axis.md")
    with open(md_out, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join("---" for _ in cols) + "|\n")
        for _, r in out.iterrows():
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    print(f"wrote {csv_out}\nwrote {md_out}")
    return out


def main():
    df = load()
    plot_metric(df, "test_spearman", "Test Spearman", "test_spearman_by_axis.png")
    plot_metric(df, "test_pearson", "Test Pearson", "test_pearson_by_axis.png")
    out = write_tables(df)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
