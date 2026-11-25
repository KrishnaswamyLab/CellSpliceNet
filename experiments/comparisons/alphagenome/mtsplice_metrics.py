"""Tissue-specific splicing metrics exactly as defined for TSplice/MTSplice
(Cheng et al., Genome Biol 2021, doi:10.1186/s13059-021-02273-7), on our GTEx/ASCOT test set.

Target: per exon e and tissue t, deviation d_et = logit(PSI_et) - logit(mean_t PSI_et).
"Variable exons": |PSI_et - mean_t PSI_et| >= 0.2 in at least one tissue, and observed in >= 10 tissues.
Metrics (reported in the paper for held-out chromosomes, 1,621 variable exons):
  per-tissue Spearman of predicted vs observed d across variable exons -> median over tissues (paper: 0.22)
  per-exon  Spearman of predicted vs observed d across tissues        -> median over exons   (paper: 0.25)
Our test set is gene-disjoint (unseen genes), i.e. also unseen exons.

usage: python mtsplice_metrics.py PRED_CSV PRED_COL [PRED_CSV PRED_COL ...]
  PRED_CSV must have event_id, neuron and the prediction column; targets come from the gene split's test.csv.
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from experiments.common.paths import split_csv
TEST = split_csv("gene", "test")
EPS = 1e-3
logit = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))

tgt = pd.read_csv(TEST, usecols=["event_id", "neuron", "PSI"]).rename(columns={"PSI": "target"})


def score(df, col):
    df = df.dropna(subset=[col])
    g = df.groupby("event_id")
    df = df[g.neuron.transform("count") >= 10]
    g = df.groupby("event_id")
    var = (df.target - g.target.transform("mean")).abs().groupby(df.event_id).max() >= 0.2
    df = df[df.event_id.map(var)]
    g = df.groupby("event_id")
    df = df.assign(d_obs=logit(df.target) - logit(g.target.transform("mean")),
                   d_pred=logit(df[col]) - logit(g[col].transform("mean")))
    per_t = df.groupby("neuron").apply(lambda x: spearmanr(x.d_obs, x.d_pred).correlation if len(x) > 10 else np.nan)
    per_e = df.groupby("event_id").apply(lambda x: spearmanr(x.d_obs, x.d_pred).correlation)
    return df.event_id.nunique(), per_t.median(), per_e.median(), (per_e > 0).mean()


for f, col in zip(sys.argv[1::2], sys.argv[2::2]):
    p = pd.read_csv(f)
    p = p[["event_id", "neuron", col]].merge(tgt, on=["event_id", "neuron"])
    n, mt, me, frac = score(p, col)
    print(f"{f.split('/')[-1]:28s} {col:10s} variable exons={n:5d}  median per-tissue rho={mt:.3f}  "
          f"median per-exon rho={me:.3f}  exons ranked right direction={frac:.1%}")
