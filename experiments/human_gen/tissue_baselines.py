"""Reference predictors for the held-out-tissue split (run with the mioflow env: needs torch).

For each test row (event e, unseen tissue t):
  event_mean   mean PSI of e over the training tissues            -> within-event rho = 0 by construction
  nn_tissue    PSI of e in the training tissue whose expression input (scatter coefficients, the
               model's own tissue input) is most similar to t (cosine); falls back to event_mean
  knn3_tissue  similarity-weighted mean over the 3 most similar training tissues that measured e
These use only training-tissue PSI and the tissues' expression inputs, i.e. the same information the
model has. Writes <out>/human_gen/baselines/cell_test_baselines.csv and prints pooled / within-event
Spearman by gene_seen.
"""
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

from experiments.human_gen.common import SCATTER, baselines_file, read_split

X = str(SCATTER) + "/"
tr = read_split("cell", "train", usecols=["event_id", "neuron", "PSI"])
va = read_split("cell", "valid", usecols=["event_id", "neuron", "PSI"])
te = read_split("cell", "test", usecols=["gene_id", "event_id", "neuron", "PSI", "gene_seen"])
# training-tissue PSI for every event (the original-test genes are only in test, so their exons have
# no training PSI: for them all baselines fall back to the global mean -> new gene AND new tissue)
allpsi = pd.concat([tr]).pivot_table(index="event_id", columns="neuron", values="PSI")
tis_tr = list(allpsi.columns); tis_te = sorted(te.neuron.unique())
emb = {t: torch.load(X + f"scatter_coeffs_{t}.pt", map_location="cpu").flatten().float().numpy() for t in tis_tr + tis_te}
E = lambda t: emb[t] / np.linalg.norm(emb[t])
sim = pd.DataFrame([[float(E(a) @ E(b)) for b in tis_tr] for a in tis_te], index=tis_te, columns=tis_tr)
print("nearest training tissue:", {t: sim.loc[t].idxmax() + f" ({sim.loc[t].max():.3f})" for t in tis_te}, flush=True)

em = allpsi.mean(1); g = tr.PSI.mean()
def pred_row(e, t, k):
    if e not in allpsi.index: return g
    row = allpsi.loc[e].dropna()
    if row.empty: return g
    s = sim.loc[t, row.index].sort_values(ascending=False).head(k)
    w = np.clip(s.values, 1e-6, None)
    return float(np.dot(w, row[s.index].values) / w.sum())
te["event_mean"] = te.event_id.map(em).fillna(g)
te["nn_tissue"] = [pred_row(e, t, 1) for e, t in zip(te.event_id, te.neuron)]
te["knn3_tissue"] = [pred_row(e, t, 3) for e, t in zip(te.event_id, te.neuron)]
te.to_csv(baselines_file("cell"), index=False)

def within(d, c):
    d = d[d.groupby("event_id").neuron.transform("count") >= 5]
    a = d.PSI - d.groupby("event_id").PSI.transform("mean"); b = d[c] - d.groupby("event_id")[c].transform("mean")
    return spearmanr(a, b).correlation if b.std() > 1e-9 else 0.0
for gs, d in te.groupby("gene_seen"):
    for c in ("event_mean", "nn_tissue", "knn3_tissue"):
        print(f"gene_seen={gs} {c:12s} n={len(d):,} pooled={spearmanr(d.PSI, d[c]).correlation:.3f} within={within(d, c):.3f}", flush=True)
