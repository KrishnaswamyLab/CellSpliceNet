"""2x2 distribution figure. Columns = {C. elegans, Human}, rows = {PSI, read depth}.
 a (worm  | PSI)         b (human | PSI)
 c (worm  | read depth)  d (human | read depth)
Read depth in matched units for both: pooled junction reads per event x cell-type
(worm nb_reads per neuron; human depth_total per tissue). Values differ in scale; units match.
Organism column headers; panels labeled a-d; no per-panel titles.
Output -> outputs/reviewer_phate/descriptives/combined_distributions_grid.png
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.common.paths import HUMAN, gtex_counts_dir, out_dir, split_csv

OUT = out_dir("reviewer", "descriptives")

# ---- load ----
wdf = pd.concat([pd.read_csv(split_csv("rows", s, "c_elegans"), usecols=["PSI", "nb_reads"])
                 for s in ["train", "valid", "test"]], ignore_index=True)
hpsi = pd.concat([pd.read_csv(split_csv("gene", s), usecols=["PSI"]) for s in ["train", "valid", "test"]],
                 ignore_index=True)
hrd = pd.read_csv(OUT / "human_read_depth.csv.gz", usecols=["depth_total"])

# ---- style ----
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight"})
# C. elegans = blue (both panels), Human = red (both panels)
C_WPSI, C_HPSI, C_WRD, C_HRD = "#4C72B0", "#C44E52", "#4C72B0", "#C44E52"
psi_bins = np.linspace(0, 1, 41)

wnb = wdf["nb_reads"].dropna(); wnb = wnb[wnb > 0].to_numpy()
hnb = hrd.loc[hrd.depth_total > 0, "depth_total"].to_numpy()
wbins = np.logspace(np.log10(wnb.min()), np.log10(wnb.max()), 50)
hbins = np.logspace(np.log10(hnb.min()), np.log10(hnb.max()), 50)

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
(ax_a, ax_b), (ax_c, ax_d) = axes

# top row: PSI
ax_a.hist(wdf.PSI, bins=psi_bins, color=C_WPSI, linewidth=0)
ax_a.set_xlabel("PSI"); ax_a.set_ylabel("samples"); ax_a.set_xlim(0, 1)
ax_b.hist(hpsi.PSI, bins=psi_bins, color=C_HPSI, linewidth=0)
ax_b.set_xlabel("PSI"); ax_b.set_ylabel("samples"); ax_b.set_xlim(0, 1)

# bottom row: read depth (matched units, log-x)
ax_c.hist(wnb, bins=wbins, color=C_WRD, alpha=0.9); ax_c.set_xscale("log")
ax_c.set_xlabel("read depth (reads per event × cell type)"); ax_c.set_ylabel("samples")
ax_c.axvline(np.median(wnb), color="black", ls="--", lw=1)
ax_d.hist(hnb, bins=hbins, color=C_HRD, alpha=0.9); ax_d.set_xscale("log")
ax_d.set_xlabel("read depth (reads per event × cell type)"); ax_d.set_ylabel("samples")
ax_d.axvline(np.median(hnb), color="black", ls="--", lw=1)

# panel labels
for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], ["a", "b", "c", "d"]):
    ax.text(-0.13, 1.16, lab, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")

fig.tight_layout(w_pad=3.0, h_pad=3.0)
# organism column headers, centered over each column
for ax, name, italic in [(ax_a, "C. elegans", True), (ax_b, "Human", False)]:
    pos = ax.get_position()
    fig.text((pos.x0 + pos.x1) / 2, 0.99, name, ha="center", va="top",
             fontsize=15, fontweight="bold", fontstyle="italic" if italic else "normal")

out = OUT / "combined_distributions_grid.png"
fig.savefig(out)
plt.close(fig)
print(f"wrote {out}")
print(f"worm: PSI n={len(wdf):,} nb_med={np.median(wnb):.0f} | "
      f"human: PSI n={len(hpsi):,} depth_total_med={np.median(hnb):.0f}")
