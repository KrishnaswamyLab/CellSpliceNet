"""Combined 2x2 distribution figure (no titles; panels labeled a-d).
 a (top-left):    worm PSI distribution - all events (SE + CE/CI controls)
 b (top-right):   worm PSI distribution - SE only (controls removed)
 c (bottom-left): worm read-depth (nb_reads) distribution
 d (bottom-right):human (GTEx) PSI distribution
Output -> outputs/reviewer_phate/descriptives/combined_distributions_2x2.png
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.common.paths import HUMAN, gtex_counts_dir, out_dir, split_csv

OUT = out_dir("reviewer", "descriptives")
OUT.mkdir(parents=True, exist_ok=True)


# ---- load ----
wdf = pd.concat([pd.read_csv(split_csv("rows", s, "c_elegans")) for s in ["train", "valid", "test"]],
                ignore_index=True)
wdf["etype"] = wdf.event_id.str.split("_").str[0]

hcols = ["gene_id", "event_id", "PSI", "neuron"]
hdf = pd.concat([pd.read_csv(split_csv("gene", s), usecols=hcols) for s in ["train", "valid", "test"]],
                ignore_index=True)

# ---- style (matches source scripts) ----
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight"})
C_ALL, C_SE, C_RD, C_HUMAN = "#4C72B0", "#55A868", "#c45a2c", "#C44E52"
psi_bins = np.linspace(0, 1, 41)

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
(ax_a, ax_b), (ax_c, ax_d) = axes

# a: worm PSI - all events
ax_a.hist(wdf.PSI, bins=psi_bins, color=C_ALL, linewidth=0)
ax_a.set_xlabel("PSI"); ax_a.set_ylabel("samples"); ax_a.set_xlim(0, 1)

# b: worm PSI - SE only
se = wdf[wdf.etype == "SE"]
ax_b.hist(se.PSI, bins=psi_bins, color=C_SE, linewidth=0)
ax_b.set_xlabel("PSI"); ax_b.set_ylabel("samples"); ax_b.set_xlim(0, 1)

# c: worm read depth
nb = wdf["nb_reads"].dropna(); nb = nb[nb > 0]
logbins = np.logspace(np.log10(nb.min()), np.log10(nb.max()), 50)
ax_c.hist(nb, bins=logbins, color=C_RD, alpha=0.9)
ax_c.set_xscale("log")
ax_c.set_xlabel("nb_reads (junction reads per event × cell-type)"); ax_c.set_ylabel("samples")
ax_c.axvline(nb.median(), color="black", ls="--", lw=1)

# d: human (GTEx) PSI
ax_d.hist(hdf.PSI, bins=psi_bins, color=C_HUMAN, linewidth=0)
ax_d.set_xlabel("PSI"); ax_d.set_ylabel("samples"); ax_d.set_xlim(0, 1)

# panel labels a-d
for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], ["a", "b", "c", "d"]):
    ax.text(-0.12, 1.12, lab, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")

fig.tight_layout(w_pad=3.0, h_pad=3.0)
out = OUT / "combined_distributions_2x2.png"
fig.savefig(out)
plt.close(fig)
print(f"wrote {out}")
print(f"worm all events: {len(wdf):,} | worm SE only: {len(se):,} | "
      f"worm nb_reads median: {nb.median():.0f} | human: {len(hdf):,}")
