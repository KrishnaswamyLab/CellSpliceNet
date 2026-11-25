"""Stage B: tissue-delta model on cached stage-A event embeddings (gene_x_cell split).

    PSI(e, t) = sigmoid( logit(base_e) + u_e . v_t + c_t )
  base_e : stage-A prediction (new exons) or measured mean over training tissues (seen exons; also
           used as the training base).  Frozen -> curriculum / stop-grad.
  u_e    : MLP of the event representation, U_MODE = emb (stage-A embedding + base logit) |
           feat (10 audit exon features + base logit) | emb+feat | free (per-event vector; seen exons only)
  v_t,c_t: free vector / offset per training tissue.  An unseen tissue gets
           v_t = sum_t' w(t,t') v_t'  with w = softmax(-||L(x_t - x_t')||^2 / T) over training tissues,
           x = tissue expression (ALL = pseudobulk PCs; SF = 493 splicing-factor means).  The kernel L is
           trained jointly: with probability P_LOO a training tissue's (v, c) is replaced by its kernel
           estimate from the other training tissues (so the same path that serves new tissues is trained).
Loss: MSE on PSI over observed (event, tissue) training cells.  Evaluated per gene_x_cell test quadrant with
the pilot_report within-event Spearman, overall and on tissue-variable events, next to the delta_probe
baselines.  Embeddings: extract_embeddings.py on the stage-A runs (hg_gene_x_cell_em_csn_noexpr_em_s*).
Outputs -> <out>/human_gen/delta_head/
"""
import os, glob, itertools
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from experiments.human_gen.common import (EVENT_FEATURES, EVENT_VARIABILITY, FC, PSEUDOBULK, SCATTER, out,
                                          read_split, split_runs, tissue_folder_map)

OUT = str(out("delta_head")) + "/"
EXD = str(SCATTER) + "/"
A2F = {v: k for k, v in tissue_folder_map().items()}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tr = read_split("gene_x_cell", "train", usecols=["event_id", "neuron", "PSI"])
va = read_split("gene_x_cell", "valid", usecols=["event_id", "neuron", "PSI"])
te = read_split("gene_x_cell", "test", usecols=["event_id", "neuron", "PSI", "gene_seen", "tissue_seen"])
te["q"] = np.select([(te.gene_seen == 1) & (te.tissue_seen == 0), (te.gene_seen == 0) & (te.tissue_seen == 1)],
                    ["Q1_new_tissue", "Q2_new_exon"], "Q3_both_new")
F = pd.read_csv(EVENT_FEATURES).set_index("event_id")
ev = pd.read_csv(EVENT_VARIABILITY, index_col=0)
TR = sorted(tr.neuron.unique()); VT = sorted(set(va.neuron) - set(TR)); TE = sorted(set(te.neuron) - set(TR))
ALLT = TR + VT + TE

# tissue representations (same construction as delta_probe.py)
PB = pd.read_csv(PSEUDOBULK, index_col=0)
PB = PB[PB[TR].mean(1) > 1.0]
pca = PCA(n_components=20, random_state=0).fit(PB[TR].T.values)
XR = {"ALL": pd.DataFrame(pca.transform(PB[ALLT].T.values), index=ALLT),
      "SF": pd.DataFrame({t: pd.read_csv(EXD + f"moments__{A2F[t]}.csv", index_col=0)["mean"].values for t in ALLT}).T}
for k in XR: X = XR[k]; XR[k] = (X - X.loc[TR].mean()) / (X.loc[TR].std() + 1e-6)

# training matrix (train exons x train tissues) and measured bases
P = tr.pivot_table(index="event_id", columns="neuron", values="PSI")[TR]
meas_base = P.mean(1)
logit = lambda p: np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99)))


class DeltaModel(nn.Module):
    def __init__(self, d_in, n_events, x_tis, k=16, rank=4, u_mode="emb"):
        super().__init__()
        self.u_mode = u_mode
        self.u = nn.Embedding(n_events, k) if u_mode == "free" else \
            nn.Sequential(nn.Linear(d_in, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, k))
        if u_mode == "free": nn.init.normal_(self.u.weight, std=0.01)
        self.v = nn.Parameter(0.01 * torch.randn(len(TR), k)); self.c = nn.Parameter(torch.zeros(len(TR)))
        self.register_buffer("x", torch.tensor(x_tis.values, dtype=torch.float32))   # all tissues, TR first
        # init sharp enough that the nearest tissue gets real weight (a flat init collapses to uniform
        # weights = event mean, seen in delta_probe kernel_ALL); log_beta = learnable inverse temperature
        self.L = nn.Parameter(0.3 * np.sqrt(20 / x_tis.shape[1]) * torch.randn(x_tis.shape[1], rank))
        self.log_beta = nn.Parameter(torch.zeros(()))

    def kernel(self, q_idx, exclude_self):
        z = self.x @ self.L; d2 = torch.cdist(z[q_idx], z[: len(TR)]) ** 2
        if exclude_self: d2 = d2.masked_fill(torch.eye(len(TR), device=d2.device, dtype=torch.bool)[q_idx], 1e9)
        return torch.softmax(-self.log_beta.exp() * d2, 1)           # |q| x |TR|

    def tissue_params(self, p_loo=0.0):
        """(v, c) for every tissue in ALLT; training tissues use their own params except a random p_loo subset."""
        W = self.kernel(torch.arange(len(ALLT), device=self.x.device), False)
        Wtr = self.kernel(torch.arange(len(TR), device=self.x.device), True)
        v_k, c_k = torch.cat([Wtr, W[len(TR):]]) @ self.v, torch.cat([Wtr, W[len(TR):]]) @ self.c
        own = torch.zeros(len(ALLT), dtype=torch.bool, device=self.x.device); own[: len(TR)] = True
        if p_loo > 0: own[: len(TR)] &= torch.rand(len(TR), device=self.x.device) >= p_loo
        v = torch.where(own[:, None], torch.cat([self.v, v_k[len(TR):]]), v_k)
        c = torch.where(own, torch.cat([self.c, c_k[len(TR):]]), c_k)
        return v, c

    def forward(self, ue_in, base_logit, p_loo=0.0):
        u = self.u(ue_in)
        v, c = self.tissue_params(p_loo)
        return torch.sigmoid(base_logit[:, None] + u @ v.T + c[None, :])   # events x ALLT


def run(emb_file, u_mode, tis_rep, seed=0, epochs=30, p_loo=0.3, var_only=False):
    torch.manual_seed(seed); np.random.seed(seed)
    z = np.load(emb_file) if emb_file else None
    if z is not None:
        E = pd.DataFrame(z["emb"].astype(np.float32), index=z["event_id"]); bp = pd.Series(z["base_pred"], index=z["event_id"])
        E = (E - E.loc[E.index.isin(P.index)].mean()) / (E.loc[E.index.isin(P.index)].std() + 1e-6)
    events = sorted(set(P.index) | set(te.event_id) | set(va.event_id))
    eidx = {e: i for i, e in enumerate(events)}
    # base: measured mean over training tissues for seen exons, stage-A prediction otherwise
    base = pd.Series(meas_base.reindex(events).values, index=events)
    if z is not None: base = base.fillna(bp.reindex(events))
    base = base.fillna(meas_base.mean())
    feats = []
    if u_mode in ("emb", "emb+feat"): feats.append(E.reindex(events).fillna(0).values)
    if u_mode in ("feat", "emb+feat"):
        f = F.reindex(events)[FC]; f = (f - f.loc[P.index].mean()) / (f.loc[P.index].std() + 1e-6); feats.append(f.fillna(0).values)
    bl = logit(base.values)
    if u_mode == "free": U_in = torch.arange(len(events))
    else: U_in = torch.tensor(np.concatenate(feats + [bl[:, None]], 1), dtype=torch.float32)
    BL = torch.tensor(bl, dtype=torch.float32)
    model = DeltaModel(U_in.shape[1] if u_mode != "free" else 0, len(events), XR[tis_rep].loc[ALLT], u_mode=u_mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tr_ev = P.index.values
    if var_only: tr_ev = tr_ev[ev.sd_all.reindex(tr_ev).fillna(0).values >= .05]
    Y = torch.tensor(P.loc[tr_ev].values, dtype=torch.float32); M = ~torch.isnan(Y); Y = torch.nan_to_num(Y)
    ti = torch.tensor([eidx[e] for e in tr_ev])
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(tr_ev)); tot = 0
        for b in perm.split(512):
            i = ti[b]
            pred = model(U_in[i].to(dev), BL[i].to(dev), p_loo)[:, : len(TR)]
            m = M[b].to(dev); loss = (((pred - Y[b].to(dev)) ** 2) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
    model.eval()
    with torch.no_grad():
        allp = []
        for b in torch.arange(len(events)).split(4096):
            allp.append(model(U_in[b].to(dev), BL[b].to(dev)).cpu())
        Pall = pd.DataFrame(torch.cat(allp).numpy(), index=events, columns=ALLT)
        Wk = pd.DataFrame(model.kernel(torch.arange(len(ALLT), device=dev), False).cpu().numpy(), index=ALLT, columns=TR)
    d = te.copy(); d["pred"] = Pall.values[[eidx[e] for e in d.event_id], [ALLT.index(t) for t in d.neuron]]
    # "+obs": for exons measured in the training tissues, transfer their measured PSI to a new tissue with
    # the learned kernel (NaN-aware weighted average) instead of using u_e . v_t
    d["pred"] = d.pred.astype(float); d["pred_obs"] = d.pred
    m = d.event_id.isin(P.index) & ~d.neuron.isin(TR)
    for t, idx in d[m].groupby("neuron").groups.items():
        v = P.reindex(d.loc[idx, "event_id"]).values; w = Wk.loc[t].values[None, :] * ~np.isnan(v)
        d.loc[idx, "pred_obs"] = np.nansum(np.nan_to_num(v) * w, 1) / np.maximum(w.sum(1), 1e-12)
    top = {t: f"{Wk.loc[t].idxmax()} ({Wk.loc[t].max():.2f})" for t in TE}
    print(f"  beta={model.log_beta.exp().item():.2f} kernel top: {top}", flush=True)
    return d, tot / len(tr_ev)


def within(d, c):
    d = d.dropna(subset=[c]); d = d[d.groupby("event_id").neuron.transform("count") >= 5]
    if d.event_id.nunique() < 20: return np.nan
    a = d.PSI - d.groupby("event_id").PSI.transform("mean"); b = d[c] - d.groupby("event_id")[c].transform("mean")
    return spearmanr(a, b).correlation if b.std() > 1e-9 else 0.0

TIERS = (("all", 0), ("sd>=0.05", .05), ("sd>=0.10", .10), ("sd>=0.20", .20))
def score(d, tag):
    rows = []
    for qn, x in d.groupby("q"):
        sd = x.event_id.map(ev.sd_all).fillna(0)
        for c, sfx in (("pred", ""), ("pred_obs", " +obs")):
            if c == "pred_obs" and qn != "Q1_new_tissue": continue
            for tier, th in TIERS:
                y = x[sd >= th]
                rows.append(dict(config=tag + sfx, quadrant=qn, tier=tier, within=within(y, c),
                                 pooled=spearmanr(y.PSI, y[c]).correlation))
    return rows

if __name__ == "__main__":
    embs = sorted(r + "/event_embeddings_best_validation_model.npz" for r in split_runs("hg", "gene_x_cell_em", "csn_noexpr_em_s", model="csn_noexpr")
                  if os.path.exists(r + "/event_embeddings_best_validation_model.npz"))
    print("embeddings:", embs, flush=True)
    configs = [("feat", None), ("free", None)] + [("emb", e) for e in embs] + [("emb+feat", e) for e in embs]
    SMOKE = os.environ.get("SMOKE") == "1"
    if SMOKE: configs, OUT = configs[:1], OUT + "smoke_"
    rows = []
    for (u_mode, e), rep, var_only in itertools.product(configs, ("ALL", "SF"), (False, True)):
        if SMOKE and var_only: continue
        tag = f"u={u_mode}{'' if e is None else '|' + e.split('_em_')[-1].split('/')[0]} tis={rep} var_only={var_only}"
        d, l = run(e, u_mode, rep, var_only=var_only, epochs=2 if SMOKE else 30)
        r = score(d, tag); rows += r
        print(tag, f"train_loss={l:.4f}", " ".join(f"{x['quadrant'][:2]}/{x['tier']}={x['within']:.3f}" for x in r if x["tier"] in ("all", "sd>=0.10")), flush=True)
        d[["event_id", "neuron", "PSI", "q", "pred", "pred_obs"]].to_csv(OUT + f"preds_{tag.replace(' ', '_').replace('|', '-').replace('=', '')}.csv.gz", index=False)
    res = pd.DataFrame(rows); res.to_csv(OUT + "delta_head_results.csv", index=False)
    tab = res.pivot_table(index=["quadrant", "config"], columns="tier", values="within").round(3)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    txt = tab.to_string() + "\n"; open(OUT + "summary.txt", "w").write(txt); print(txt, flush=True)
