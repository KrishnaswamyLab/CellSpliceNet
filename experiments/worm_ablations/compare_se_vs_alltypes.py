"""Compare the SE-only-trained anchor vs the all-types-trained anchor on the SAME
SE test events. Tests the hypothesis that CE/CI constitutive controls hurt
cell-type (across-neuron) generalization.

usage: compare_se_vs_alltypes.py <SE-only-trained preds csv> <all-types-trained preds csv>
(per-row predictions of the two anchors on the SE test rows, e.g. from dump_preds.sh / src/eval.py;
the all-types file is filtered to SE events)

Metric of interest: across-neuron Spearman on SE events with >=5 neurons (the
cell-type axis). Also pooled SE Spearman and per-event pred-range vs target-range.
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from experiments.worm_ablations.common import read_preds

SE_PRED, ALL_PRED = sys.argv[1], sys.argv[2]


def across_neuron(df, min_n=5):
    rhos, prng, trng = [], [], []
    for ev, g in df.groupby("event_id"):
        if g.neuron.nunique() >= min_n:
            if g.target.std() > 0 and g.pred.std() > 0:
                rhos.append(spearmanr(g.pred, g.target).statistic)
            prng.append(g.pred.max() - g.pred.min())
            trng.append(g.target.max() - g.target.min())
    return (np.nanmean(rhos), len(rhos), np.median(prng), np.median(trng))


se = read_preds(SE_PRED)
al = read_preds(ALL_PRED)
al = al[al.event_id.str.startswith("SE")].copy()

# restrict both to the SAME (event,neuron) keys for a fair head-to-head
key = lambda d: set(zip(d.event_id, d.neuron))
common = key(se) & key(al)
se["k"] = list(zip(se.event_id, se.neuron)); al["k"] = list(zip(al.event_id, al.neuron))
se = se[se.k.isin(common)].drop(columns="k")
al = al[al.k.isin(common)].drop(columns="k")
print(f"common SE (event,neuron) samples: {len(common)}  "
      f"(SE-only file {len(read_preds(SE_PRED))}, all-types SE {len(al)})\n")

for name, d in [("ALL-TYPES-trained (eval on SE)", al), ("SE-ONLY-trained (eval on SE)", se)]:
    pooled = spearmanr(d.pred, d.target).statistic
    mae = np.abs(d.pred - d.target).mean()
    an, n_ev, pr, tr = across_neuron(d)
    print(f"=== {name} ===")
    print(f"  pooled SE Spearman : {pooled:.4f}   MAE: {mae:.4f}")
    print(f"  ACROSS-NEURON Spearman (>=5 neurons, n={n_ev} events): {an:.4f}")
    print(f"  per-event pred-range median {pr:.3f}  vs  target-range median {tr:.3f}\n")
