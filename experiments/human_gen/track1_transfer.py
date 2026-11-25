"""Track 1 decisive test: can an expression -> PSI model trained on GTEx SAMPLES predict unseen tissues from their
expression alone, better than the nearest-tissue lookup?

Data: <data_root>/human/processed/donor (processing/human/donor_data.py).  Tissues follow gene_x_cell: 34 train / 5 valid / 10 test.
Donors are split 80/20 (seed 0).  Train = train tissues x train donors.  Valid = valid tissues x test donors.
Test = test tissues x test donors  (new tissue AND new donor; no sQTL / donor leakage).
Exon identity is seen (free per-exon parameters) -> this is the Q1 'new tissue, seen exon' setting; exon
generalization comes later by predicting u_e from sequence.
Expression x_s: log1p-CPM, z-scored on training samples; feature sets SF (493 splicing factors) and ALL (all
expressed genes), each reduced to 50 PCs fitted on training samples.
Methods (tissue-level prediction = mean over the held-out tissue's test-donor samples):
  lookup_SF / lookup_ALL : nearest training tissue by pseudobulk expression -> its pooled training PSI
  knn20_*                : each test sample -> mean PSI of its 20 nearest training samples (expression PCs)
  ridge_*                : multi-output ridge, PSI ~ PCs (missing PSI imputed with the exon's training mean)
  lowrank_*              : PSI = sigmoid(b_e + u_e . f(x_s)), f = MLP(PCs) -> 32, binomial loss on read counts
Metrics: within-event Spearman across the 10 test tissues (as pilot_report; truth = pooled test-donor PSI),
by ASCOT variability tier; and donor-level: per (exon, tissue) Pearson of predicted vs observed PSI across
test-donor samples of a held-out tissue (exons with >=20 reads in >=20 samples), median.
Outputs -> <out>/human_gen/track1/
"""
import os
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from experiments.human_gen.common import (DONOR, EVENT_FEATURES, EVENT_VARIABILITY, SCATTER, SS_WINDOWS, all_rows,
                                          out, read_split, split_runs)

DD = str(DONOR) + "/"
OUT = str(out("track1")) + "/"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", 0))          # donor split + model init
METHODS = os.environ.get("METHODS", "lookup,knn,ridge,lowrank,lowrank_tissue").split(",")
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

M = pd.read_csv(DD + "meta_samples.csv"); EV = pd.read_csv(DD + "meta_events.csv").event_id.values
GN = pd.read_csv(DD + "meta_genes.csv"); is_sf = GN.is_sf.values
ev = pd.read_csv(EVENT_VARIABILITY, index_col=0)
TR = sorted(read_split("gene_x_cell", "train", usecols=["neuron"]).neuron.unique())
VA_all = sorted(read_split("gene_x_cell", "valid", usecols=["neuron"]).neuron.unique()); VT = [t for t in VA_all if t not in TR]
TE = sorted(set(read_split("gene_x_cell", "test", usecols=["neuron"]).neuron.unique()) - set(TR))
# robustness: TISSUE_FOLD=k (0-4) replaces the combo tissue split by a rotation in which every tissue is held out once
# (5 folds of 10/10/10/10/9 test tissues; 5 valid tissues drawn from the rest).  -1 = combo split.
FOLD = int(os.environ.get("TISSUE_FOLD", -1))
TAG = f"seed{SEED}" + (f"_fold{FOLD}" if FOLD >= 0 else "") + os.environ.get("RUN_TAG", "")
if FOLD >= 0:
    perm = list(np.random.default_rng(123).permutation(sorted(M.tissue.unique())))
    TE = sorted(perm[FOLD::5]); rest = [t for t in perm if t not in TE]; VT = sorted(rest[:5]); TR = sorted(rest[5:])
    print(f"fold {FOLD}: test tissues {TE}\n  valid {VT}", flush=True)
donors = M.donor.unique(); test_donors = set(rng.choice(donors, int(0.2 * len(donors)), replace=False))
M["dtest"] = M.donor.isin(test_donors)
M["role"] = np.select([M.tissue.isin(TR) & ~M.dtest, M.tissue.isin(VT) & M.dtest, M.tissue.isin(TE) & M.dtest],
                      ["train", "valid", "test"], "unused")
print(M.groupby("role").agg(samples=("sample", "size"), tissues=("tissue", "nunique"), donors=("donor", "nunique")), flush=True)

# load arrays in meta order, keep used samples
PSI, TOT, X = [], [], []
for f, d in M.groupby("folder", sort=False):
    z = np.load(DD + f + ".npz"); PSI.append(z["psi"][d.row.values]); TOT.append(z["tot"][d.row.values]); X.append(z["expr"][d.row.values])
order = np.concatenate([d.index.values for _, d in M.groupby("folder", sort=False)])
M = M.loc[order].reset_index(drop=True)
PSI = np.concatenate(PSI).astype(np.float32); TOT = np.concatenate(TOT).astype(np.float32); X = np.concatenate(X).astype(np.float32)
if FOLD >= 0:
    # re-select exons with this fold's TRAINING tissues only (ASCOT SD >= 0.05), so held-out tissues never choose exons
    asc = all_rows(["event_id", "neuron", "PSI"]).pivot_table(index="event_id", columns="neuron", values="PSI")
    sd_tr = asc.reindex(EV)[TR].std(1)
    keep = (sd_tr >= 0.05).values & (asc.reindex(EV)[TR].notna().sum(1) >= 5).values
    EV = EV[keep]; PSI = PSI[:, keep]; TOT = TOT[:, keep]
    print(f"fold {FOLD}: exons kept {keep.sum():,} of {len(keep):,}", flush=True)
# all reads: binomial loss + pooled tissue-level PSI; sample-level PSI (ridge / kNN / donor metric) needs >= 20 reads
INC = np.nan_to_num(PSI) * TOT; TOTm = np.where(np.isnan(PSI), 0, TOT)
PSI[TOT < 20] = np.nan
tr = (M.role == "train").values
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-3
Xz = (X - mu) / sd
FEATS = {}
for nm, cols in (("SF", is_sf), ("ALL", np.ones_like(is_sf, dtype=bool))):
    pca = PCA(50, random_state=0).fit(Xz[tr][:, cols]); Z = pca.transform(Xz[:, cols])
    FEATS[nm] = (Z - Z[tr].mean(0)) / Z[tr].std(0)
    print(f"{nm}: genes {cols.sum()}, PCA var explained {pca.explained_variance_ratio_.sum():.2f}", flush=True)
exon_mean = np.nanmean(PSI[tr], 0)

# ---------------- tissue-level truth for held-out tissues (pooled over test-donor samples) ----------------
def pooled(mask):
    inc = INC[mask].sum(0); tot = TOTm[mask].sum(0)
    return np.where(tot >= 20, inc / np.maximum(tot, 1), np.nan)
def truth(role):
    T = sorted(M.tissue[M.role == role].unique())
    return T, pd.DataFrame({t: pooled(((M.role == role) & (M.tissue == t)).values) for t in T}, index=EV)
TRUTH = {r: truth(r) for r in ("valid", "test")}
POOL_TR = pd.DataFrame({t: pooled(((M.role == "train") & (M.tissue == t)).values) for t in TR}, index=EV)

def within(true_df, pred_df, th=0.0):
    t = true_df.stack().rename("PSI").reset_index(); t.columns = ["event_id", "tissue", "PSI"]
    p = pred_df.stack().rename("pred").reset_index(); p.columns = ["event_id", "tissue", "pred"]
    d = t.merge(p, on=["event_id", "tissue"]).dropna()
    d = d[d.event_id.map(ev.sd_all).fillna(0) >= th]
    d = d[d.groupby("event_id").tissue.transform("count") >= 5]
    a = d.PSI - d.groupby("event_id").PSI.transform("mean"); b = d.pred - d.groupby("event_id").pred.transform("mean")
    return spearmanr(a, b).correlation, d.event_id.nunique()

def donor_level(Pred, role):
    """median over (exon, held-out tissue) of Pearson(pred, obs) across that tissue's test-donor samples"""
    rs = []
    for t in sorted(M.tissue[M.role == role].unique()):
        m = ((M.role == role) & (M.tissue == t)).values
        Y = PSI[m]; P = Pred[m]; ok = (~np.isnan(Y)).sum(0) >= 20
        for j in np.where(ok)[0]:
            y = Y[:, j]; k = ~np.isnan(y)
            if np.std(y[k]) > 1e-6 and np.std(P[k, j]) > 1e-9: rs.append(np.corrcoef(y[k], P[k, j])[0, 1])
    return float(np.median(rs)) if rs else np.nan

results, per_tissue = [], []
def tissue_breakdown(name, true_df, pred_df):
    """per held-out tissue: Spearman of true vs predicted deviation (from each exon's mean over the held-out tissues)"""
    ok = true_df.notna().sum(1) >= 5
    a = true_df[ok].sub(true_df[ok].mean(1), axis=0); b = pred_df.loc[ok, true_df.columns]
    b = b.sub(b.mean(1), axis=0)
    for t in true_df.columns:
        m = a[t].notna() & b[t].notna()
        per_tissue.append(dict(method=name, tissue=t, events=int(m.sum()), rho=spearmanr(a[t][m], b[t][m]).correlation))

def report(name, tissue_pred, sample_pred=None):
    tissue_breakdown(name, TRUTH["test"][1], tissue_pred["test"])
    for role in ("valid", "test"):
        T, tru = TRUTH[role]
        row = dict(method=name, split=role)
        for tag, th in (("all", 0), ("sd>=0.10", .1), ("sd>=0.20", .2)):
            row[f"within_{tag}"], row[f"events_{tag}"] = within(tru, tissue_pred[role], th)
        row["donor_level_r"] = donor_level(sample_pred, role) if sample_pred is not None else np.nan
        results.append(row); print(row, flush=True)

def tissue_avg(Pred, role):
    T = TRUTH[role][0]
    return pd.DataFrame({t: np.nanmean(Pred[((M.role == role) & (M.tissue == t)).values], 0) for t in T}, index=EV)

# ---------------- lookup (nonparametric, tissue level) ----------------
for nm in (("SF", "ALL") if "lookup" in METHODS else ()):
    Z = FEATS[nm]; cen = {t: Z[((M.role == "train") & (M.tissue == t)).values].mean(0) for t in TR}
    tp = {}
    for role in ("valid", "test"):
        T = TRUTH[role][0]; cols = {}
        for t in T:
            q = Z[((M.role == role) & (M.tissue == t)).values].mean(0)
            nn_t = min(TR, key=lambda s: np.linalg.norm(cen[s] - q)); cols[t] = POOL_TR[nn_t].values
        tp[role] = pd.DataFrame(cols, index=EV)
    report(f"lookup_{nm}", tp)

# ---------------- sample kNN ----------------
for nm in (("SF", "ALL") if "knn" in METHODS else ()):
    Z = FEATS[nm]; nnb = NearestNeighbors(n_neighbors=20).fit(Z[tr])
    Ytr = np.where(np.isnan(PSI[tr]), exon_mean, PSI[tr])
    Pred = np.full_like(PSI, np.nan)
    ev_rows = np.where(M.role.isin(["valid", "test"]))[0]
    _, idx = nnb.kneighbors(Z[ev_rows]); Pred[ev_rows] = Ytr[idx].mean(1)
    report(f"knn20_{nm}", {r: tissue_avg(Pred, r) for r in ("valid", "test")}, Pred)

# ---------------- ridge ----------------
Ytr = np.where(np.isnan(PSI[tr]), exon_mean, PSI[tr]); ybar = Ytr.mean(0)
for nm in (("SF", "ALL") if "ridge" in METHODS else ()):
    for lam in (10.0, 1000.0):
        Z = FEATS[nm]; A = Z[tr]
        Wr = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr - ybar))
        Pred = np.clip(Z @ Wr + ybar, 0, 1)
        report(f"ridge{int(lam)}_{nm}", {r: tissue_avg(Pred, r) for r in ("valid", "test")}, Pred)

# ---------------- low-rank neural, binomial loss ----------------
class LowRank(nn.Module):
    def __init__(self, d, n_ev, k=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, k))
        self.u = nn.Parameter(0.01 * torch.randn(n_ev, k))
        m = np.clip(np.nan_to_num(exon_mean, nan=0.5), .02, .98)
        self.b = nn.Parameter(torch.tensor(np.log(m / (1 - m)), dtype=torch.float32))
    def forward(self, x): return self.f(x) @ self.u.T + self.b

for nm in (("SF", "ALL") if "lowrank" in METHODS else ()):
    Z = torch.tensor(FEATS[nm], dtype=torch.float32)
    model = LowRank(Z.shape[1], len(EV)).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tri = np.where(tr)[0]; best, best_state = -1, None
    for ep in range(60):
        model.train()
        for b in np.array_split(rng.permutation(tri), max(1, len(tri) // 256)):
            lg = model(Z[b].to(dev)); n = torch.tensor(TOTm[b], device=dev); k = torch.tensor(INC[b], device=dev)
            # binomial NLL per read, averaged over reads (caps very deep exons' weight via sqrt)
            w = n.sqrt() / (n + 1e-6)
            loss = -((k * nn.functional.logsigmoid(lg) + (n - k) * nn.functional.logsigmoid(-lg)) * w).sum() / (n * w).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4:
            model.eval()
            with torch.no_grad(): Pred = torch.sigmoid(model(Z.to(dev))).cpu().numpy()
            score = within(TRUTH["valid"][1], tissue_avg(Pred, "valid"))[0]
            if score > best: best, best_state = score, {k_: v.clone() for k_, v in model.state_dict().items()}
            print(f"  lowrank_{nm} ep {ep + 1} loss {loss.item():.4f} valid within {score:.3f}", flush=True)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): Pred = torch.sigmoid(model(Z.to(dev))).cpu().numpy()
    report(f"lowrank_{nm}", {r: tissue_avg(Pred, r) for r in ("valid", "test")}, Pred)

# ---------------- CONTROL: same low-rank model trained on TISSUE-LEVEL data (one row per training tissue) ----------
# Separates "per-sample training" from "model class": inputs are tissue-level (pseudobulk of the expression PCs,
# or the CSN scatter tensor used by the current model), targets are reads pooled over the tissue's training donors.
if "lowrank_tissue" in METHODS or "exon_arms" in METHODS:
    SCD = str(SCATTER) + "/"
    ALLT = TR + TRUTH["valid"][0] + TRUTH["test"][0]
    sc = np.stack([torch.load(SCD + f"scatter_coeffs_{t}.pt", map_location="cpu").flatten().numpy() for t in ALLT])
    sc = (sc - sc[: len(TR)].mean(0)) / (sc[: len(TR)].std(0) + 1e-6)
    pca_sc = PCA(min(20, len(TR) - 1), random_state=0).fit(sc[: len(TR)])
    Zsc = pca_sc.transform(sc); Zsc = (Zsc - Zsc[: len(TR)].mean(0)) / (Zsc[: len(TR)].std(0) + 1e-6)
    def tissue_rows(role_of):  # tissue -> mask of samples whose expression / reads define that tissue
        return {t: ((M.role == role_of(t)) & (M.tissue == t)).values for t in ALLT}
    masks = tissue_rows(lambda t: "train" if t in TR else ("valid" if t in TRUTH["valid"][0] else "test"))
    INPUTS = {f"PB_{nm}": np.stack([FEATS[nm][masks[t]].mean(0) for t in ALLT]) for nm in ("SF", "ALL")}
    INPUTS["CSNscatter"] = Zsc
    K_tr = torch.tensor(np.stack([INC[masks[t]].sum(0) for t in TR]), dtype=torch.float32, device=dev)
    N_tr = torch.tensor(np.stack([TOTm[masks[t]].sum(0) for t in TR]), dtype=torch.float32, device=dev)
    for nm, Xin in (INPUTS.items() if "lowrank_tissue" in METHODS else []):
        Xt = torch.tensor(Xin, dtype=torch.float32, device=dev)
        model = LowRank(Xt.shape[1], len(EV)).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state = -1, None
        for step in range(3000):
            model.train(); lg = model(Xt[: len(TR)]); w = N_tr.sqrt() / (N_tr + 1e-6)
            loss = -((K_tr * nn.functional.logsigmoid(lg) + (N_tr - K_tr) * nn.functional.logsigmoid(-lg)) * w).sum() / (N_tr * w).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 100 == 99:
                model.eval()
                with torch.no_grad(): P = torch.sigmoid(model(Xt)).cpu().numpy()
                vt = TRUTH["valid"][0]
                score = within(TRUTH["valid"][1], pd.DataFrame({t: P[ALLT.index(t)] for t in vt}, index=EV))[0]
                if score > best: best, best_state = score, {k_: v.clone() for k_, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad(): P = torch.sigmoid(model(Xt)).cpu().numpy()
        print(f"  lowrank_tissue_{nm} best valid within {best:.3f}", flush=True)
        report(f"lowrank_tissue_{nm}", {r: pd.DataFrame({t: P[ALLT.index(t)] for t in TRUTH[r][0]}, index=EV) for r in ("valid", "test")})

# ---------------- EXON-FEATURE arm: exon side from the 10 audit sequence features (inductive over exons) ----------
# Exon partition = combo chromosome split (train exons / chr2 valid / chrom-test exons).  Quadrants:
#   Q1 train exons x held-out tissues | Q2 test exons x training tissues (held-out donors) | Q3 test exons x held-out tissues
# Baselines on the same rows: per-tissue offset and per-tissue GBM(features -> deviation), transferred to held-out
# tissues through the nearest training tissue (lookup_SF).
if "exon_feat" in METHODS or "exon_arms" in METHODS:
    from sklearn.ensemble import HistGradientBoostingRegressor
    FC = ["log_exon", "log_up", "log_dn", "len_mod3", "gc_exon", "gc_up", "gc_dn", "maxent3", "maxent5", "ag_up"]
    part = pd.concat([read_split("gene_x_cell_em", k, usecols=["event_id", "part"])
                      for k in ("train", "valid", "test")]).set_index("event_id").part.reindex(EV).values
    Fe = pd.read_csv(EVENT_FEATURES).set_index("event_id").reindex(EV)[FC]
    Fe = ((Fe - Fe[part == "train"].mean()) / (Fe[part == "train"].std() + 1e-6)).fillna(0).values.astype(np.float32)
    exT, exV, exE = (part == "train"), (part == "valid"), (part == "test")
    print(f"exon_feat: exons train {exT.sum()} valid {exV.sum()} test {exE.sum()}", flush=True)
    # Q2 truth: training tissues x held-out donors (role 'unused' rows of training tissues)
    q2mask = {t: ((M.tissue == t) & M.dtest).values for t in TR}
    TRUTH_Q2 = pd.DataFrame({t: pooled(q2mask[t]) for t in TR}, index=EV)
    def sub(df, m): return df.loc[EV[m]]
    def qscores(name, P_q1, P_q2, P_q3):
        for q, tru, pr, m in (("Q1_new_tissue", TRUTH["test"][1], P_q1, exT), ("Q2_new_exon", TRUTH_Q2, P_q2, exE),
                              ("Q3_both_new", TRUTH["test"][1], P_q3, exE)):
            if pr is None: continue
            row = dict(method=name, split=q)
            for tag, th in (("all", 0), ("sd>=0.10", .1), ("sd>=0.20", .2)):
                row[f"within_{tag}"], row[f"events_{tag}"] = within(sub(tru, m), sub(pr, m), th)
            results.append(row); print(row, flush=True)
    # nearest training tissue by SF pseudobulk (same rule as lookup_SF)
    Zs = FEATS["SF"]; cen = {t: Zs[((M.role == "train") & (M.tissue == t)).values].mean(0) for t in TR}
    nn_of = {t: min(TR, key=lambda s: np.linalg.norm(cen[s] - Zs[((M.role == "test") & (M.tissue == t)).values].mean(0)))
             for t in TRUTH["test"][0]}
    # baselines
    D_tr = POOL_TR.sub(POOL_TR.mean(1), axis=0)                          # deviations on training tissues
    off = D_tr[exT].mean(0)
    G = pd.DataFrame(index=EV, columns=TR, dtype=float)
    for t in TR:
        y = D_tr[t].values; ok = exT & ~np.isnan(y)
        G[t] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Fe[ok], y[ok]).predict(Fe)
    Toff = pd.DataFrame(np.tile(off.values, (len(EV), 1)), index=EV, columns=TR)
    qscores("BASE_offset", None, Toff, pd.DataFrame({t: Toff[nn_of[t]] for t in TRUTH["test"][0]}))
    qscores("BASE_gbm_per_tissue", None, G, pd.DataFrame({t: G[nn_of[t]] for t in TRUTH["test"][0]}))

    class FeatLowRank(nn.Module):
        def __init__(self, d, k=32):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, k))
            self.u = nn.Sequential(nn.Linear(len(FC), 128), nn.GELU(), nn.Linear(128, k))
            self.b = nn.Sequential(nn.Linear(len(FC), 128), nn.GELU(), nn.Linear(128, 1))
            self.F = torch.tensor(Fe, device=dev)
        def forward(self, x): return self.f(x) @ self.u(self.F).T + self.b(self.F).T
    colT = torch.tensor(exT, device=dev)
    def binom(lg, k, n):
        w = n.sqrt() / (n + 1e-6) * colT
        return -((k * nn.functional.logsigmoid(lg) + (n - k) * nn.functional.logsigmoid(-lg)) * w).sum() / ((n * w).sum() + 1e-6)
    def valid_score(P_tissue):  # chr2 exons x valid tissues
        return within(sub(TRUTH["valid"][1], exV), sub(P_tissue, exV))[0]

    # (a) sample-level training
    for nm in (("SF", "ALL") if "exon_feat" in METHODS else ()):
        Z = torch.tensor(FEATS[nm], dtype=torch.float32)
        model = FeatLowRank(Z.shape[1]).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        tri = np.where(tr)[0]; best, best_state = -1, None
        for ep in range(60):
            model.train()
            for b in np.array_split(rng.permutation(tri), max(1, len(tri) // 256)):
                loss = binom(model(Z[b].to(dev)), torch.tensor(INC[b], device=dev), torch.tensor(TOTm[b], device=dev))
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 5 == 4:
                model.eval()
                with torch.no_grad(): Pred = torch.sigmoid(model(Z.to(dev))).cpu().numpy()
                s = valid_score(tissue_avg(Pred, "valid"))
                if s > best: best, best_state = s, {k_: v.clone() for k_, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad(): Pred = torch.sigmoid(model(Z.to(dev))).cpu().numpy()
        P2 = pd.DataFrame({t: np.nanmean(Pred[q2mask[t]], 0) for t in TR}, index=EV)
        P13 = tissue_avg(Pred, "test")
        print(f"  featlowrank_sample_{nm} best valid {best:.3f}", flush=True)
        qscores(f"featlowrank_sample_{nm}", P13, P2, P13)

    # (b) tissue-level training (one row per training tissue), inputs PB_SF or CSN scatter
    if "lowrank_tissue" in METHODS and "exon_feat" in METHODS:
        for nm, Xin in INPUTS.items():
            if nm == "PB_ALL": continue
            Xt = torch.tensor(Xin, dtype=torch.float32, device=dev)
            model = FeatLowRank(Xt.shape[1]).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best, best_state = -1, None
            for step in range(3000):
                model.train(); loss = binom(model(Xt[: len(TR)]), K_tr, N_tr)
                opt.zero_grad(); loss.backward(); opt.step()
                if step % 100 == 99:
                    model.eval()
                    with torch.no_grad(): P = torch.sigmoid(model(Xt)).cpu().numpy()
                    s = valid_score(pd.DataFrame({t: P[ALLT.index(t)] for t in TRUTH["valid"][0]}, index=EV))
                    if s > best: best, best_state = s, {k_: v.clone() for k_, v in model.state_dict().items()}
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad(): P = torch.sigmoid(model(Xt)).cpu().numpy()
            P2 = pd.DataFrame({t: P[ALLT.index(t)] for t in TR}, index=EV)
            P13 = pd.DataFrame({t: P[ALLT.index(t)] for t in TRUTH["test"][0]}, index=EV)
            print(f"  featlowrank_tissue_{nm} best valid {best:.3f}", flush=True)
            qscores(f"featlowrank_tissue_{nm}", P13, P2, P13)

# ---------------- EXON ARMS: generic exon encoder x tissue side (exon-minibatch training) ----------------
# PSI(row, e) = sigmoid( b_e + u_e . f(x_row) ), (u_e, b_e) = MLP(exon encoder(e)), trained on training exons only.
#   EXON_ENC    comma list of: feat | conv | conv+feat | emb | emb+feat | kmer | kmer+feat
#               feat = 10 audit features; conv = dilated CNN on single-nt windows around the 4 splice sites
#               (ss_windows.npz, 4 x 350 nt); emb = CSN stage-A embedding (csn_noexpr on combo_em, EMB_SEED)
#               kmer = 1-5-mer frequencies in 8 exon/intron regions of the same windows (+ ridge exon-mean baselines)
#   TISSUE_SIDE comma list of: tissue_CSNscatter (rows = tissues, scatter-tensor PCs) | sample_SF (rows = samples)
# Also scores exon-level ranking: 'Q2_between' = Spearman over test exons of mean predicted vs mean true PSI
# across training tissues (held-out donors); baseline GBM(features) -> exon mean.
if "exon_arms" in METHODS:
    import glob
    ENCS = os.environ.get("EXON_ENC", "feat,conv,conv+feat").split(",")
    SIDES = os.environ.get("TISSUE_SIDE", "tissue_CSNscatter,sample_SF").split(",")
    logsig = nn.functional.logsigmoid
    WIN = np.load(SS_WINDOWS, allow_pickle=True)
    widx = pd.Series(np.arange(len(WIN["event_id"])), index=WIN["event_id"]).reindex(EV)
    SEQ = torch.tensor(WIN["seq"][widx.fillna(0).astype(int).values].astype(np.int64)); SEQ[torch.tensor(widx.isna().values)] = 4
    FEt = torch.tensor(Fe)
    EMB = None
    embf = [r + "/event_embeddings_best_validation_model.npz" for r in
            split_runs("hg", "gene_x_cell_em", "csn_noexpr_em_s" + os.environ.get("EMB_SEED", "1"), model="csn_noexpr")
            if r.endswith("_s" + os.environ.get("EMB_SEED", "1"))
            and os.path.exists(r + "/event_embeddings_best_validation_model.npz")]
    if embf:
        z = np.load(embf[0], allow_pickle=True); Em = pd.DataFrame(z["emb"].astype(np.float32), index=z["event_id"]).reindex(EV)
        print(f"exon_arms: CSN embeddings {embf[0].split('/')[-2]} dim {Em.shape[1]}, missing {int(Em.isna().all(1).sum())}", flush=True)
        Em = (Em - Em[exT].mean()) / (Em[exT].std() + 1e-6); EMB = torch.tensor(Em.fillna(0).values)
    ENCS = [e for e in ENCS if "emb" not in e or EMB is not None]
    # kmer = 1..5-mer frequencies in 8 regions (exon / intron side of each splice-site segment, transcript order):
    #   A: upstream exon 100 | intron 250   B: intron 300 | target exon 50   C: target exon 50 | intron 300
    #   D: intron 250 | downstream exon 100    (k-mers containing N are dropped)
    KM = None
    if any("kmer" in e for e in ENCS):
        S = WIN["seq"][widx.fillna(0).astype(int).values].astype(np.int64); S[widx.isna().values] = 4
        cuts = [0, 100, 350, 650, 700, 750, 1050, 1300, 1400]
        blocks = []
        for kk in range(1, 6):
            L_ = S.shape[1] - kk + 1; code = np.zeros((S.shape[0], L_), np.int64); bad = np.zeros((S.shape[0], L_), bool)
            for j in range(kk):
                code = code * 4 + np.minimum(S[:, j: j + L_], 3); bad |= S[:, j: j + L_] == 4
            for a, b in zip(cuts[:-1], cuts[1:]):
                c = code[:, a: b - kk + 1]; ok = ~bad[:, a: b - kk + 1]
                rows = np.repeat(np.arange(len(S)), ok.sum(1))
                H = np.zeros((len(S), 4 ** kk), np.float32); np.add.at(H, (rows, c[ok]), 1.0)
                blocks.append(H / np.maximum(H.sum(1, keepdims=True), 1))
        Km = np.concatenate(blocks, 1)
        Km = (Km - Km[exT].mean(0)) / (Km[exT].std(0) + 1e-3)
        KM = torch.tensor(Km.astype(np.float32)); print(f"exon_arms: kmer features {KM.shape}", flush=True)
        # exon-level ridge baseline on k-mers (+ features) -> exon mean PSI
        from sklearn.linear_model import RidgeCV
    print(f"exon_arms: encoders {ENCS} sides {SIDES}", flush=True)

    def exon_level(name, P_q2):
        m = exE; t_ = TRUTH_Q2.loc[EV[m]].mean(1); p_ = P_q2.loc[EV[m]].mean(1); ok = t_.notna() & p_.notna()
        row = dict(method=name, split="Q2_between", within_all=spearmanr(t_[ok], p_[ok]).correlation, events_all=int(ok.sum()))
        results.append(row); print(row, flush=True)
    gm = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0)
    ym = POOL_TR.mean(1).values; okm = exT & ~np.isnan(ym); gm.fit(Fe[okm], ym[okm])
    exon_level("BASE_gbm_mean", pd.DataFrame(np.tile(gm.predict(Fe)[:, None], (1, len(TR))), index=EV, columns=TR))
    if KM is not None:
        for nm_, X_ in (("kmer", Km), ("kmer+feat", np.concatenate([Km, Fe], 1))):
            rg = RidgeCV(alphas=np.logspace(1, 5, 9)).fit(X_[okm], ym[okm])
            print(f"  ridge {nm_} alpha {rg.alpha_:.0f}", flush=True)
            exon_level(f"BASE_ridge_{nm_}_mean", pd.DataFrame(np.tile(rg.predict(X_)[:, None], (1, len(TR))), index=EV, columns=TR))

    class ConvEnc(nn.Module):
        def __init__(self, out, ch=64):
            super().__init__()
            self.stem = nn.Conv1d(4, ch, 11, padding=5)
            self.blocks = nn.ModuleList([nn.Sequential(nn.GroupNorm(8, ch), nn.GELU(), nn.Conv1d(ch, ch, 11, padding=5 * d, dilation=d))
                                         for d in (1, 2, 4, 8)])
            self.head = nn.Sequential(nn.Linear(ch * 4 * 2, 256), nn.GELU(), nn.Linear(256, out))
        def forward(self, s):
            x = self.stem(nn.functional.one_hot(s, 5)[..., :4].float().transpose(1, 2))
            for b in self.blocks: x = x + b(x)
            seg = x.reshape(x.shape[0], x.shape[1], 4, 350)                   # per splice-site segment
            return self.head(torch.cat([seg.mean(-1), seg.amax(-1)], 1).flatten(1))

    class ArmModel(nn.Module):
        def __init__(self, enc, d_t, k=32):
            super().__init__()
            self.enc = enc; parts = enc.split("+")
            self.conv = ConvEnc(128) if "conv" in parts else None
            d = (128 if self.conv is not None else 0) + (Fe.shape[1] if "feat" in parts else 0) + (EMB.shape[1] if "emb" in parts else 0)
            self.kmer = nn.Sequential(nn.Dropout(0.2), nn.Linear(KM.shape[1], 128)) if "kmer" in parts else None
            d += 128 if self.kmer is not None else 0
            self.E = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, k + 1))
            self.f = nn.Sequential(nn.Linear(d_t, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, k))
        def exon(self, idx):
            parts = self.enc.split("+"); h = []
            if self.conv is not None: h.append(self.conv(SEQ[idx].to(dev)))
            if "feat" in parts: h.append(FEt[idx].to(dev))
            if "emb" in parts: h.append(EMB[idx].to(dev))
            if self.kmer is not None: h.append(self.kmer(KM[idx].to(dev)))
            return self.E(torch.cat(h, 1))
        def forward(self, x, idx):
            ub = self.exon(idx); return self.f(x) @ ub[:, :-1].T + ub[:, -1][None, :]
        @torch.no_grad()
        def predict(self, x):
            UB = torch.cat([self.exon(c).float() for c in torch.arange(len(EV)).split(1024)])
            return torch.sigmoid(self.f(x).float() @ UB[:, :-1].T + UB[:, -1][None, :]).cpu().numpy()

    trE = torch.tensor(np.where(exT)[0]); tri = np.where(tr)[0]
    XT = torch.tensor(INPUTS["CSNscatter"], dtype=torch.float32, device=dev)
    ZS = torch.tensor(FEATS["SF"], dtype=torch.float32, device=dev)
    vmask = M.role.values == "valid"
    for side, enc in [(s, e) for s in SIDES for e in ENCS]:
        torch.manual_seed(SEED)
        tissue_side = side.startswith("tissue")
        model = ArmModel(enc, (XT if tissue_side else ZS).shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state = -1, None; steps = int(os.environ.get("ARM_STEPS", 6000))
        for step in range(steps):
            model.train(); ei = trE[torch.randint(len(trE), (512,))]
            if tissue_side:
                x = XT[: len(TR)]; k = K_tr[:, ei.to(dev)]; n = N_tr[:, ei.to(dev)]
            else:
                b = np.sort(rng.choice(tri, 256, replace=False)); x = ZS[b]
                k = torch.tensor(INC[b][:, ei.numpy()], device=dev); n = torch.tensor(TOTm[b][:, ei.numpy()], device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lg = model(x, ei).float()
            w = n.sqrt() / (n + 1e-6)
            loss = -((k * logsig(lg) + (n - k) * logsig(-lg)) * w).sum() / ((n * w).sum() + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 500 == 499:
                model.eval()
                if tissue_side:
                    P = model.predict(XT); Pv = pd.DataFrame({t: P[ALLT.index(t)] for t in TRUTH["valid"][0]}, index=EV)
                else:
                    Pfull = np.full((len(M), len(EV)), np.nan, np.float32); Pfull[vmask] = model.predict(ZS[torch.tensor(vmask, device=dev)])
                    Pv = tissue_avg(Pfull, "valid")
                s = valid_score(Pv)
                print(f"    {side} {enc} step {step + 1} loss {loss.item():.4f} valid {s:.3f}", flush=True)
                if s > best: best, best_state = s, {k_: v.clone() for k_, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
        if tissue_side:
            P = model.predict(XT)
            P2 = pd.DataFrame({t: P[ALLT.index(t)] for t in TR}, index=EV)
            P13 = pd.DataFrame({t: P[ALLT.index(t)] for t in TRUTH["test"][0]}, index=EV)
        else:
            Pfull = np.full((len(M), len(EV)), np.nan, np.float32)
            need = (M.role.values == "test") | (M.dtest.values & M.tissue.isin(TR).values)
            Pfull[need] = model.predict(ZS[torch.tensor(need, device=dev)])
            P2 = pd.DataFrame({t: np.nanmean(Pfull[q2mask[t]], 0) for t in TR}, index=EV); P13 = tissue_avg(Pfull, "test")
        name = f"arm_{side}_{enc}"
        print(f"  {name} best valid {best:.3f}", flush=True)
        qscores(name, P13, P2, P13); exon_level(name, P2)
        del model; torch.cuda.empty_cache()

res = pd.DataFrame(results); res["seed"] = SEED; res["fold"] = FOLD; res.to_csv(OUT + f"track1_results_{TAG}.csv", index=False)
pt = pd.DataFrame(per_tissue)
if len(pt): pt["seed"] = SEED; pt["fold"] = FOLD; pt.to_csv(OUT + f"track1_per_tissue_{TAG}.csv", index=False)
pd.set_option("display.width", 250)
txt = res.round(3).to_string(index=False) + "\n\n" + (pt.pivot_table(index="tissue", columns="method", values="rho").round(3).to_string() if len(pt) else "") + "\n"
open(OUT + f"summary_{TAG}.txt", "w").write(txt); print(txt, flush=True)
