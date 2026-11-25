"""Minimal worm dataset descriptive: PSI histogram (top) + counts table (bottom).
Two versions: all event types (SE+CE+CI) and SE-only (CE/CI controls removed).
Outputs -> outputs/reviewer_phate/descriptives/
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

df = pd.concat([pd.read_csv(split_csv("rows", s, "c_elegans")) for s in ["train", "valid", "test"]])
df["etype"] = df.event_id.str.split("_").str[0]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})


def make(d, title, fname, color, ctype_label="Neuron types"):
    counts = [
        (f"Samples (event × {ctype_label.split()[0].lower().rstrip('s')})", f"{len(d):,}"),
        ("Unique events",            f"{d.event_id.nunique():,}"),
        ("Unique genes",             f"{d.gene_id.nunique():,}"),
        (ctype_label,                f"{d.neuron.nunique()}"),
    ]
    if "etype" in d.columns and d.etype.nunique() > 1:
        comp = d.etype.value_counts()
        names = {"SE": "Skipped exon (SE)", "CE": "Cassette exon (CE)", "CI": "Cassette intron (CI)"}
        for t in ["SE", "CE", "CI"]:
            if t in comp:
                counts.append((f"   {names[t]}", f"{comp[t]:,}"))

    fig = plt.figure(figsize=(5.2, 5.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.05], hspace=0.55)

    ax = fig.add_subplot(gs[0])
    ax.hist(d.PSI, bins=np.linspace(0, 1, 41), color=color, linewidth=0)
    ax.set_xlabel("PSI"); ax.set_ylabel("samples"); ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=11)

    axt = fig.add_subplot(gs[1]); axt.axis("off")
    tbl = axt.table(cellText=counts, colLabels=["Dataset composition", "count"],
                    cellLoc="left", colLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#f0f0f0")
        if c == 1:
            cell.set_text_props(ha="right")

    fig.savefig(OUT / fname)
    plt.close(fig)
    print(f"wrote {OUT/fname}  ({len(d):,} samples)")


make(df, "Worm PSI distribution — all events (SE + CE/CI controls)",
     "psi_table_with_CXCI.png", "#4C72B0")
make(df[df.etype == "SE"], "Worm PSI distribution — SE only (controls removed)",
     "psi_table_SEonly.png", "#55A868")

# ---- Human (GTEx): single event class (splice junctions), one version ----
hcols = ["gene_id", "event_id", "PSI", "neuron"]
hdf = pd.concat([pd.read_csv(split_csv("gene", s), usecols=hcols) for s in ["train", "valid", "test"]])
make(hdf, "Human (GTEx) PSI distribution", "psi_table_human.png", "#C44E52",
     ctype_label="Tissues")
