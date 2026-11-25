"""Embedding-free probe of the planned base + tissue-delta design on the combo split.

Question: how much within-event (tissue-specific) signal can simple inputs recover in each
test quadrant of the gene_x_cell split?  These are the numbers a CSN delta head has to beat.
  Q1 new tissue  (gene_seen=1, tissue_seen=0): seen exon, unseen tissue
      event_mean (within 0 by construction); nn_tissue by expression (two tissue representations:
      SF = per-tissue mean of the 493 splicing factors the model gets (moments__*.csv),
      ALL = pseudobulk log1p-CPM of all genes (processed/expression/tissue_pseudobulk.csv.gz), PCA on
      training tissues); kernel = learned
      expression metric + softmax over training tissues, fitted leave-one-tissue-out on the training
      tissues; ORACLE_nn = nearest training tissue chosen with the test labels (reference only).
  Q2 new exon    (gene_seen=0, tissue_seen=1): unseen exon, seen tissue
      tissue_offset (mean deviation of the tissue); per-tissue GBM on the 10 exon features -> deviation.
  Q3 both new    (0, 0): per-tissue GBM / offsets of training tissues, combined by the Q1 tissue
      kernel (or nearest tissue).
Scored with the pilot_report within-event Spearman, overall and on tissue-variable events
(ASCOT SD over tissues, processed/event_variability.csv).
Outputs -> <out>/human_gen/delta_probe/
"""
import os, glob
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from experiments.human_gen.common import (EVENT_FEATURES, EVENT_VARIABILITY, FC, PSEUDOBULK, SCATTER, out,
                                          read_split, tissue_folder_map)

OUT = str(out("delta_probe")) + "/"
EXD = str(SCATTER) + "/"
torch.manual_seed(0); np.random.seed(0)
A2F = {v: k for k, v in tissue_folder_map().items()}

tr = read_split("gene_x_cell", "train", usecols=["event_id", "neuron", "PSI"])
te = read_split("gene_x_cell", "test", usecols=["event_id", "neuron", "PSI", "gene_seen", "tissue_seen"])
F = pd.read_csv(EVENT_FEATURES).set_index("event_id")
ev = pd.read_csv(EVENT_VARIABILITY, index_col=0)
TR = sorted(tr.neuron.unique()); TE = sorted(te.neuron[te.tissue_seen == 0].unique())
print(f"train tissues {len(TR)}, test tissues {len(TE)}", flush=True)

# ---------------- tissue representations ----------------
def sf_vec(t):
    m = pd.read_csv(EXD + f"moments__{A2F[t]}.csv", index_col=0)
    return m["mean"].values.astype(float)

PB = pd.read_csv(PSEUDOBULK, index_col=0)[TR + TE]
PB = PB[PB[TR].mean(1) > 1.0]
pca = PCA(n_components=20, random_state=0).fit(PB[TR].T.values)
X_all = pd.DataFrame(pca.transform(PB[TR + TE].T.values), index=TR + TE)
X_sf = pd.DataFrame({t: sf_vec(t) for t in TR + TE}).T
def zs(X): mu, sd = X.loc[TR].mean(), X.loc[TR].std() + 1e-6; return (X - mu) / sd
X_all, X_sf = zs(X_all), zs(X_sf)

# ---------------- training matrices ----------------
P = tr.pivot_table(index="event_id", columns="neuron", values="PSI")[TR]
base = P.mean(1); D = P.sub(base, axis=0)                     # deviations on training tissues
offset = D.mean(0)                                              # per-tissue offset

def cos_sim(X, q, keys):
    a = X.loc[q].values; B = X.loc[keys].values
    return B @ a / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + 1e-9)

# learned tissue kernel: w(q, t') = softmax(-||L(x_q - x_t')||^2), fitted leave-one-tissue-out
def fit_kernel(X, rank=4, steps=600, wd=1e-2):
    x = torch.tensor(X.loc[TR].values, dtype=torch.float32)
    Dt = torch.tensor(D.fillna(0).values, dtype=torch.float32); M = torch.tensor(D.notna().values, dtype=torch.float32)
    L = torch.nn.Parameter(0.1 * torch.randn(x.shape[1], rank))
    opt = torch.optim.Adam([L], lr=3e-2, weight_decay=wd)
    eye = torch.eye(len(TR), dtype=torch.bool)
    for _ in range(steps):
        z = x @ L; d2 = torch.cdist(z, z) ** 2
        W = torch.softmax((-d2).masked_fill(eye, -1e9), dim=1)        # q x t'
        num = (Dt * M) @ W.T; den = M @ W.T + 1e-6                    # events x q
        pred = num / den
        loss = (((pred - Dt) ** 2) * M).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return L.detach()

def kernel_weights(X, L, q):
    z = torch.tensor(X.loc[TR].values, dtype=torch.float32) @ L
    zq = torch.tensor(X.loc[[q]].values, dtype=torch.float32) @ L
    return pd.Series(torch.softmax(-(torch.cdist(zq, z) ** 2)[0], 0).numpy(), index=TR)

K = {"SF": (X_sf, fit_kernel(X_sf)), "ALL": (X_all, fit_kernel(X_all))}
W = {}  # test tissue -> method -> weights over TR
for t in TE:
    W[t] = {}
    for nm, (X, L) in K.items():
        s = pd.Series(cos_sim(X, t, TR), index=TR)
        W[t][f"nn_{nm}"] = (s == s.max()).astype(float)
        W[t][f"kernel_{nm}"] = kernel_weights(X, L, t)
    print(f"  {t}: nn_SF={W[t]['nn_SF'].idxmax()}  nn_ALL={W[t]['nn_ALL'].idxmax()}  "
          f"kernel_ALL top={W[t]['kernel_ALL'].idxmax()} ({W[t]['kernel_ALL'].max():.2f})", flush=True)

def wavg(Mat, w):  # weighted average over training-tissue columns, ignoring NaN
    v = Mat[w.index].values; m = ~np.isnan(v); ww = w.values[None, :] * m
    return np.where(ww.sum(1) > 0, np.nansum(np.nan_to_num(v) * ww, 1) / np.maximum(ww.sum(1), 1e-12), np.nan)

# ---------------- per-tissue GBM on exon features -> deviation ----------------
G = pd.DataFrame(index=te.event_id.unique(), columns=TR, dtype=float)
ids = [e for e in G.index if e in F.index]
for t in TR:
    y = D[t].dropna(); y = y[y.index.isin(F.index)]
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(F.loc[y.index, FC], y)
    G.loc[ids, t] = m.predict(F.loc[ids, FC])
print("per-tissue GBM done", flush=True)

# ---------------- predictions per quadrant ----------------
te["q"] = np.select([(te.gene_seen == 1) & (te.tissue_seen == 0), (te.gene_seen == 0) & (te.tissue_seen == 1)],
                    ["Q1_new_tissue", "Q2_new_exon"], "Q3_both_new")
preds = {}
q1 = te[te.q == "Q1_new_tissue"].copy()
for meth in ["nn_SF", "kernel_SF", "nn_ALL", "kernel_ALL"]:
    q1[meth] = np.nan
    for t, idx in q1.groupby("neuron").groups.items():
        q1.loc[idx, meth] = wavg(P.reindex(q1.loc[idx, "event_id"]), W[t][meth])
# oracle nearest training tissue, picked with test labels (reference only)
q1["ORACLE_nn"] = np.nan
for t, d in q1.groupby("neuron"):
    dd = d.set_index("event_id").PSI - base.reindex(d.event_id).values
    c = D.reindex(dd.index).corrwith(dd); q1.loc[d.index, "ORACLE_nn"] = P.reindex(d.event_id)[c.idxmax()].values
q1["event_mean"] = base.reindex(q1.event_id).values
preds["Q1_new_tissue"] = (q1, ["event_mean", "nn_SF", "kernel_SF", "nn_ALL", "kernel_ALL", "ORACLE_nn"])

q2 = te[te.q == "Q2_new_exon"].copy()
q2["tissue_offset"] = q2.neuron.map(offset)
q2["gbm_dev_per_tissue"] = [G.at[e, t] if e in G.index else np.nan for e, t in zip(q2.event_id, q2.neuron)]
preds["Q2_new_exon"] = (q2, ["tissue_offset", "gbm_dev_per_tissue"])

q3 = te[te.q == "Q3_both_new"].copy()
for meth in ["nn_ALL", "kernel_ALL", "kernel_SF"]:
    q3[f"offset@{meth}"] = q3.neuron.map({t: float((offset * W[t][meth]).sum() / W[t][meth].sum()) for t in TE})
    q3[f"gbm@{meth}"] = np.nan
    for t, idx in q3.groupby("neuron").groups.items():
        q3.loc[idx, f"gbm@{meth}"] = wavg(G.reindex(q3.loc[idx, "event_id"]), W[t][meth])
preds["Q3_both_new"] = (q3, [c for c in q3.columns if c.startswith(("offset@", "gbm@"))])

# ---------------- scoring ----------------
def within(d, c):
    d = d.dropna(subset=[c]); d = d[d.groupby("event_id").neuron.transform("count") >= 5]
    if d.event_id.nunique() < 20: return np.nan, 0
    a = d.PSI - d.groupby("event_id").PSI.transform("mean"); b = d[c] - d.groupby("event_id")[c].transform("mean")
    return (spearmanr(a, b).correlation if b.std() > 1e-9 else 0.0), d.event_id.nunique()

rows = []
for qn, (d, cols) in preds.items():
    sd = d.event_id.map(ev.sd_all).fillna(0)
    for tier, th in (("all", 0), ("sd>=0.05", .05), ("sd>=0.10", .10), ("sd>=0.20", .20)):
        x = d[sd >= th]
        for c in cols:
            w, n = within(x, c); rows.append(dict(quadrant=qn, method=c, tier=tier, within=w, events=n))
    d.to_csv(OUT + f"preds_{qn}.csv.gz", index=False)
res = pd.DataFrame(rows); res.to_csv(OUT + "delta_probe_results.csv", index=False)
tab = res.pivot_table(index=["quadrant", "method"], columns="tier", values="within").round(3)
ne = res.pivot_table(index="quadrant", columns="tier", values="events", aggfunc="max")
txt = tab.to_string() + "\n\nevents (>=5 tissues) per tier:\n" + ne.to_string() + "\n"
open(OUT + "summary.txt", "w").write(txt); print(txt, flush=True)
