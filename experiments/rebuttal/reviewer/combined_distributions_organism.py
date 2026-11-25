"""2x2 distribution figure: rows = {worm, human}, columns = {PSI, read depth}.
 a (worm  | PSI):        worm PSI
 b (worm  | read depth): worm nb_reads (junction reads per event x neuron), log-x
 c (human | PSI):        human (GTEx) PSI
 d (human | read depth): human per-sample junction read depth (reconstructed), log-x
Two versions written:
  combined_distributions_organism_overlay.png  -> panel a stacked SE vs CE/CI controls
  combined_distributions_organism_allevents.png -> panel a all events, single colour
No titles; panels labeled a-d. Caption explains content.
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
wdf = pd.concat([pd.read_csv(split_csv("rows", s, "c_elegans")) for s in ["train", "valid", "test"]],
                ignore_index=True)
wdf["etype"] = wdf.event_id.str.split("_").str[0]

hpsi = pd.concat([pd.read_csv(split_csv("gene", s), usecols=["PSI"]) for s in ["train", "valid", "test"]],
                 ignore_index=True)
hrd = pd.read_csv(OUT / "human_read_depth.csv.gz")

# ---- style ----
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight"})
C_ALL, C_SE, C_CTRL = "#4C72B0", "#55A868", "#bdbdbd"   # all / SE events / CE-CI controls
C_RD_W, C_HUMAN, C_RD_H = "#c45a2c", "#C44E52", "#8172B3"
psi_bins = np.linspace(0, 1, 41)

# shared data
se_psi = wdf.loc[wdf.etype == "SE", "PSI"].to_numpy()
ctrl_psi = wdf.loc[wdf.etype != "SE", "PSI"].to_numpy()
all_psi = wdf["PSI"].to_numpy()
nb = wdf["nb_reads"].dropna(); nb = nb[nb > 0].to_numpy()
b_bins = np.logspace(np.log10(nb.min()), np.log10(nb.max()), 50)
hd = hrd.loc[hrd.depth_per_sample > 0, "depth_per_sample"].to_numpy()
d_bins = np.logspace(np.log10(hd.min()), np.log10(hd.max()), 50)


def build(mode):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # a: worm PSI
    if mode == "overlay":
        ax_a.hist([se_psi, ctrl_psi], bins=psi_bins, stacked=True,
                  color=[C_SE, C_CTRL], linewidth=0,
                  label=[f"SE events (n={len(se_psi):,})",
                         f"CE/CI controls (n={len(ctrl_psi):,})"])
        ax_a.legend(frameon=False, fontsize=8.5, loc="upper center")
    else:  # all events, single colour
        ax_a.hist(all_psi, bins=psi_bins, color=C_ALL, linewidth=0)
    ax_a.set_xlabel("PSI"); ax_a.set_ylabel("samples"); ax_a.set_xlim(0, 1)

    # b: worm read depth (per event x neuron)
    ax_b.hist(nb, bins=b_bins, color=C_RD_W, alpha=0.9)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("read depth (reads per event × neuron)"); ax_b.set_ylabel("samples")
    ax_b.axvline(np.median(nb), color="black", ls="--", lw=1)

    # c: human PSI
    ax_c.hist(hpsi.PSI, bins=psi_bins, color=C_HUMAN, linewidth=0)
    ax_c.set_xlabel("PSI"); ax_c.set_ylabel("samples"); ax_c.set_xlim(0, 1)

    # d: human read depth (per-sample), log-x
    ax_d.hist(hd, bins=d_bins, color=C_RD_H, alpha=0.9)
    ax_d.set_xscale("log")
    ax_d.set_xlabel("read depth (reads per event × tissue, per sample)")
    ax_d.set_ylabel("samples")
    ax_d.axvline(np.median(hd), color="black", ls="--", lw=1)

    for ax, lab in zip([ax_a, ax_b, ax_c, ax_d], ["a", "b", "c", "d"]):
        ax.text(-0.12, 1.12, lab, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="top", ha="left")

    fig.tight_layout(w_pad=3.0, h_pad=3.0)
    out = OUT / f"combined_distributions_organism_{mode}.png"
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


build("overlay")
build("allevents")
print(f"worm SE={len(se_psi):,} ctrl={len(ctrl_psi):,} all={len(all_psi):,} nb_med={np.median(nb):.0f} | "
      f"human PSI rows={len(hpsi):,} depth/sample median={np.median(hd):.1f}")
