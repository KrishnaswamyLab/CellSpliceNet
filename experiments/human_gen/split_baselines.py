"""Feature baselines on any human split (fit on that split's training file only).
  feat_gbm       gradient boosting on the 10 audit sequence features -> event-mean PSI
  tissue_offset  training-set tissue mean deviation (within-event reference)
  gbm_plus_off   sum of the two
usage: python experiments/human_gen/split_baselines.py <split>   e.g. gene_k8, chrom_k8, gene_x_cell
       (writes <out>/human_gen/baselines/<split>_test_feature_baselines.csv)
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.human_gen.common import EVENT_FEATURES, FC, baselines_file, read_split

SPLIT = sys.argv[1]
F = pd.read_csv(EVENT_FEATURES).set_index("event_id")
tr = read_split(SPLIT, "train", usecols=["event_id", "neuron", "PSI"])
te = read_split(SPLIT, "test", usecols=["gene_id", "event_id", "neuron", "PSI"])
em = tr.groupby("event_id").PSI.mean()
gbm = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0).fit(F.loc[em.index, FC], em)
ids = te.event_id.unique()
te["feat_gbm"] = te.event_id.map(pd.Series(gbm.predict(F.loc[ids, FC]), index=ids))
tr["dev"] = tr.PSI - tr.groupby("event_id").PSI.transform("mean")
te["tissue_offset"] = te.neuron.map(tr.groupby("neuron").dev.mean()).fillna(0)
te["gbm_plus_off"] = te.feat_gbm + te.tissue_offset
te.to_csv(baselines_file(SPLIT), index=False)
def within(d, c):
    d = d[d.groupby("event_id").neuron.transform("count") >= 5]
    a = d.PSI - d.groupby("event_id").PSI.transform("mean"); b = d[c] - d.groupby("event_id")[c].transform("mean")
    return spearmanr(a, b).correlation if b.std() > 1e-9 else 0.0
for c in ("feat_gbm", "tissue_offset", "gbm_plus_off"):
    print(f"{SPLIT} {c:14s} pooled={spearmanr(te.PSI, te[c]).correlation:.3f} within={within(te, c):.3f}", flush=True)
