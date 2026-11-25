"""Dataset descriptives for the supplement (reviewer response).

Worm (C. elegans, processed_v02/5Dec2024_EventType) and Human (GTEx).
Reports: rows / unique events / unique genes / cell types, event-type breakdown,
PSI distribution, read-depth (nb_reads) distribution + QC notes.

Outputs (outputs/reviewer_phate/descriptives/):
  descriptives_table.md, descriptives_table.csv
  worm_descriptives.png, human_descriptives.png
"""
import os, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.common.paths import HUMAN, gtex_counts_dir, out_dir, split_csv

OUT = out_dir("reviewer", "descriptives")
OUT.mkdir(parents=True, exist_ok=True)

WORM_FILES = {s: split_csv("rows", s, "c_elegans") for s in ["train", "valid", "test"]}
GTEX_FILES = {s: str(split_csv("gene", s)) for s in ["train", "valid", "test"]}

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "figure.dpi": 150, "savefig.bbox": "tight",
})

EVENT_NAMES = {"SE": "Skipped exon", "CE": "Cassette exon", "CI": "Cassette intron",
               "GT": "Splice junction"}


def load(files, cols):
    parts = {}
    for s, f in files.items():
        parts[s] = pd.read_csv(f, usecols=lambda c: c in cols)
    return parts


def split_stats(parts, organism):
    rows = []
    for s, d in parts.items():
        etypes = d.event_id.astype(str).str.split("_").str[0].value_counts().to_dict()
        rows.append({
            "organism": organism, "split": s, "rows": len(d),
            "unique_events": d.event_id.nunique(), "unique_genes": d.gene_id.nunique(),
            "cell_types": d.neuron.nunique(),
            "event_types": "; ".join(f"{EVENT_NAMES.get(k,k)}={v}" for k, v in etypes.items()),
        })
    alld = pd.concat(parts.values())
    rows.append({
        "organism": organism, "split": "ALL", "rows": len(alld),
        "unique_events": alld.event_id.nunique(), "unique_genes": alld.gene_id.nunique(),
        "cell_types": alld.neuron.nunique(),
        "event_types": "; ".join(f"{EVENT_NAMES.get(k,k)}={v}" for k, v in
                                 alld.event_id.astype(str).str.split("_").str[0].value_counts().items()),
    })
    return rows, alld


# ---------------- load ----------------
cols = ["event_id", "gene_id", "neuron", "PSI", "nb_reads"]
worm_parts = load(WORM_FILES, cols)
gtex_parts = load(GTEX_FILES, cols)
worm_rows, worm_all = split_stats(worm_parts, "C. elegans (worm)")
gtex_rows, gtex_all = split_stats(gtex_parts, "Human (GTEx)")

table = pd.DataFrame(worm_rows + gtex_rows)
table.to_csv(OUT / "descriptives_table.csv", index=False)
def to_md(df):
    cols = list(df.columns)
    out = "| " + " | ".join(cols) + " |\n"
    out += "| " + " | ".join("---" for _ in cols) + " |\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return out

with open(OUT / "descriptives_table.md", "w") as f:
    f.write("# Dataset descriptives\n\n")
    f.write(to_md(table))
    f.write("\n\n## Read depth / QC\n")
    f.write(f"- Worm nb_reads (reads supporting each event-in-cell-type PSI): "
            f"median {worm_all.nb_reads.median():.1f}, "
            f"IQR [{worm_all.nb_reads.quantile(.25):.1f}, {worm_all.nb_reads.quantile(.75):.1f}], "
            f"min {worm_all.nb_reads.min():.2f}, max {worm_all.nb_reads.max():.0f}. "
            f"{(worm_all.nb_reads>=10).mean()*100:.1f}% of samples have >=10 reads.\n")
    f.write(f"- Human (GTEx) nb_reads column is a placeholder (all = "
            f"{gtex_all.nb_reads.iloc[0]:g}); read-depth/coverage QC was applied upstream "
            f"during GTEx PSI quantification and is not retained in the processed table.\n")
print(table.to_string(index=False))


# ---------------- worm figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
# (a) PSI distribution, stacked by event type
etype = worm_all.event_id.astype(str).str.split("_").str[0]
colors = {"SE": "#4C72B0", "CE": "#DD8452", "CI": "#55A868"}
bins = np.linspace(0, 1, 41)
bottom = np.zeros(len(bins) - 1)
for t in ["SE", "CE", "CI"]:
    h, _ = np.histogram(worm_all.PSI[etype == t], bins=bins)
    ax[0].bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge",
              color=colors[t], label=EVENT_NAMES[t], linewidth=0)
    bottom += h
ax[0].set(xlabel="PSI", ylabel="samples (event × cell type)", title="a  PSI distribution")
ax[0].legend(frameon=False, fontsize=7)
# (b) read depth
rd = worm_all.nb_reads[worm_all.nb_reads > 0]
ax[1].hist(np.log10(rd), bins=50, color="#8172B3", linewidth=0)
med = worm_all.nb_reads.median()
ax[1].axvline(np.log10(med), color="k", ls="--", lw=1)
ax[1].set(xlabel="reads / sample (log10)", ylabel="samples",
          title="b  Read depth")
ax[1].text(np.log10(med), ax[1].get_ylim()[1]*0.92, f" median {med:.0f}", fontsize=7)
# (c) events per gene
epg = worm_all.groupby("gene_id").event_id.nunique()
ax[2].hist(epg, bins=range(1, epg.max() + 2), color="#937860", align="left", linewidth=0)
ax[2].set(xlabel="AS events per gene", ylabel="genes", title="c  Events per gene")
fig.suptitle(f"C. elegans alternative-splicing dataset  "
             f"(n={len(worm_all):,} samples · {worm_all.event_id.nunique():,} events · "
             f"{worm_all.gene_id.nunique():,} genes · {worm_all.neuron.nunique()} neuron types)",
             fontsize=9, y=1.04)
fig.tight_layout()
fig.savefig(OUT / "worm_descriptives.png")
plt.close(fig)


# ---------------- human figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
ax[0].hist(gtex_all.PSI, bins=np.linspace(0, 1, 41), color="#4C72B0", linewidth=0)
ax[0].set(xlabel="PSI", ylabel="samples (event × tissue)", title="a  PSI distribution")
epg = gtex_all.groupby("gene_id").event_id.nunique()
ax[1].hist(epg, bins=range(1, min(epg.max(), 30) + 2), color="#937860", align="left", linewidth=0)
ax[1].set(xlabel="AS events per gene", ylabel="genes", title="b  Events per gene")
# events per tissue (coverage)
ept = gtex_all.groupby("neuron").event_id.nunique().sort_values()
ax[2].barh(range(len(ept)), ept.values, color="#55A868", height=0.8)
ax[2].set(xlabel="events measured", title="c  Events per tissue (n=49)")
ax[2].set_yticks([])
fig.suptitle(f"Human GTEx alternative-splicing dataset  "
             f"(n={len(gtex_all):,} samples · {gtex_all.event_id.nunique():,} events · "
             f"{gtex_all.gene_id.nunique():,} genes · {gtex_all.neuron.nunique()} tissues)",
             fontsize=9, y=1.04)
fig.tight_layout()
fig.savefig(OUT / "human_descriptives.png")
plt.close(fig)
print(f"\nwrote figures + table to {OUT}")
