"""Morning report for the overnight human pilot.

For every run dir matching hg_{gene_k8,chrom_k8,cell_k8}_* (or the old names orig8 / chrom8 / tissue8)
with preds_{final_model,best_validation_model}.csv:
  pooled / between-event / within-event Spearman (events with >= 5 tissues), Pearson, mean bias.
Baselines are scored on exactly the same (event, tissue) rows:
  gene_k8 / chrom_k8 : feat_gbm, tissue_offset, gbm_plus_off   (split_baselines.py, fit on that split)
  cell_k8            : event_mean, nn_tissue, knn3_tissue      (tissue_baselines.py; gene_seen rows only,
                       plus CSN on new-gene rows separately)
Writes <out>/human_gen/pilot_report.csv and prints a compact table (mean over seeds).
"""
import os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

from experiments.human_gen.common import baselines_file, out_dir, split_names, split_runs

def within(d, c):
    d = d[d.groupby("event_id").neuron.transform("count") >= 5]
    if d.empty: return np.nan
    a = d.PSI - d.groupby("event_id").PSI.transform("mean"); b = d[c] - d.groupby("event_id")[c].transform("mean")
    return spearmanr(a, b).correlation if b.std() > 1e-9 else 0.0
def metr(d, c):
    d = d.dropna(subset=[c]); m = d.groupby("event_id")[["PSI", c]].mean()
    return dict(n=len(d), events=d.event_id.nunique(), pooled=spearmanr(d.PSI, d[c]).correlation,
                between=spearmanr(m.PSI, m[c]).correlation, within=within(d, c),
                pearson=pearsonr(d.PSI, d[c])[0], bias=float((d[c] - d.PSI).mean()))

BASE = {"gene_k8": (baselines_file("gene_k8"), ["feat_gbm", "tissue_offset", "gbm_plus_off"]),
        "chrom_k8": (baselines_file("chrom_k8"), ["feat_gbm", "tissue_offset", "gbm_plus_off"]),
        "cell_k8": (baselines_file("cell"), ["event_mean", "nn_tissue", "knn3_tissue"])}
rows = []
for split, (bf, bcols) in BASE.items():
    base = pd.read_csv(bf) if os.path.exists(bf) else None
    runs = split_runs("hg", split)
    names = "|".join(split_names(split))
    evaluated = None
    for run in runs:
        m = re.search(rf"hg_(?:{names})_(.+)_s(\d+)$", run); model, seed = m.group(1), int(m.group(2))
        for ck in ("final_model", "best_validation_model"):
            f = f"{run}/preds_{ck}.csv"
            if not os.path.exists(f): continue
            p = pd.read_csv(f).rename(columns={"psi_target": "PSI", "psi_pred": "csn"})
            evaluated = p[["event_id", "neuron"]]
            # PSI_HEAD=twohead runs also carry exon_pred / dev_pred: score each head on its own
            heads = [("csn", "")] + [(c, f"[{c.split('_')[0]}]") for c in ("exon_pred", "dev_pred") if c in p.columns]
            if split == "cell_k8" and base is not None:
                p = p.merge(base[["event_id", "neuron", "gene_seen"]], on=["event_id", "neuron"], how="left")
                for gs, d in p.groupby("gene_seen"):
                    for c, sfx in heads:
                        rows.append(dict(split=split, subset=f"gene_seen={int(gs)}", model=model + sfx, seed=seed, ckpt=ck, **metr(d, c)))
            else:
                for c, sfx in heads:
                    rows.append(dict(split=split, subset="all", model=model + sfx, seed=seed, ckpt=ck, **metr(p, c)))
    if base is not None and evaluated is not None:
        b = base.merge(evaluated.drop_duplicates(), on=["event_id", "neuron"])
        groups = b.groupby("gene_seen") if split == "cell_k8" else [("all", b)]
        for gs, d in groups:
            for c in bcols:
                rows.append(dict(split=split, subset=(f"gene_seen={int(gs)}" if split == "cell_k8" else "all"),
                                 model="BASELINE:" + c, seed=0, ckpt="-", **metr(d, c)))
r = pd.DataFrame(rows)
r.to_csv(out_dir("human_gen") / "pilot_report.csv", index=False)
pd.set_option("display.width", 220)
agg = r.groupby(["split", "subset", "model", "ckpt"])[["pooled", "between", "within", "pearson", "bias", "n"]].agg(
    lambda s: f"{s.mean():.3f}" + (f"±{s.std():.3f}" if len(s) > 1 else ""))
print(agg.to_string())
