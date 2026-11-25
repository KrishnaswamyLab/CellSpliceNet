"""Plot top (high-attention) vs bottom (low-attention) vs random perturbation.

Faithfulness signature: if attention is meaningful, perturbing the TOP-attended
tokens should hurt most, RANDOM less, and BOTTOM least. One panel per op
(noise / shuffle / knockout); bars = pooled-Spearman drop from clean (0.909).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
from experiments.common.paths import out_dir
# usage: plot_high_low_random.py <perturb_faithfulness.py results csv> [out.png]
CSV = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else str(out_dir("worm_ablations", "figs") / "perturb_high_low_random.png")

df = pd.read_csv(CSV)
clean = df[df.modality == "clean"].iloc[0]
clean_pooled = clean.spearman

# core single-modality + all; keep only top/bottom/random modes
modes = ["top", "random", "bottom"]            # high -> random -> low
mode_label = {"top": "high (top-20%)", "random": "random-20%", "bottom": "low (bottom-20%)"}
mode_color = {"top": "#c0392b", "random": "#7f8c8d", "bottom": "#2980b9"}
modalities = ["sequence", "roi", "structure", "expression", "all"]
ops = ["noise", "shuffle", "knockout"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
x = np.arange(len(modalities))
w = 0.26

for ax, op in zip(axes, ops):
    sub = df[(df.op == op) & (df["mode"].isin(modes))]
    for i, mode in enumerate(modes):
        vals = []
        for m in modalities:
            row = sub[(sub.modality == m) & (sub["mode"] == mode)]
            vals.append(row.drop_pooled.iloc[0] if len(row) else np.nan)
        ax.bar(x + (i - 1) * w, vals, w, label=mode_label[mode], color=mode_color[mode])
    ax.set_title(f"op = {op}")
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, rotation=30, ha="right")
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_xlabel("modality")

axes[0].set_ylabel("pooled-Spearman drop from clean (0.909)")
axes[0].legend(title="perturbed tokens", loc="upper left", framealpha=0.9)
fig.suptitle(
    "Attention faithfulness: high vs low vs random token perturbation (seed-1, n=6512)",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
