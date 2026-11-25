"""Worm analogue of track1: is neuron expression sufficient to (1) tell neuron types apart and (2) predict splicing
in UNSEEN neurons, better than copying the nearest training neuron?

Data (<data_root>/c_elegans): events.csv = per (event, neuron, bulk replicate) nb_reads + PSI for SE/CE/CI events,
50 neurons, 2-6 replicates each; scRNA counts (CeNGEN, 33.7k cells x 22.5k genes, cell name = <neuron>_<i>);
CSN expression input = processed/expression/scatter_243/scatter_coeffs_<neuron>.pt ([243, 11, 243]).
PSI and expression come from different assays, so there is no per-sample pairing: cells are noisy samples of their type.

Part 1 (fold-free) -> part1.txt
  a. cosine similarity between neurons' CSN scatter tensors (raw and centred across neurons); same for human tissues
  b. cell -> neuron classification from SF genes vs all (HVG) genes (5-fold CV, kNN15 and logistic, balanced accuracy)
  c. split-half pseudobulk identification: half A of a neuron's cells matched to its own half B among 50 (SF / ALL)
  d. PSI identification: replicate half A profile matched to own half B; split-half within-event ceiling (all neurons)
  e. Mantel: neuron-pair expression distance vs PSI-profile distance
Part 2 (per fold; FOLDS env, -1 = frozen 'cell' split, 0-4 = rotation of 10 held-out neurons, 5 valid, 35 train)
  within-event Spearman on held-out neurons (truth = reads pooled over that neuron's replicates), by event type
  (all/SE/CE/CI) and training-neuron SD tier; methods:
    ceiling_splithalf    replicate half A vs half B of the held-out neurons (reliability bound)
    random_neuron        random training neuron's PSI (mean over 20 draws)
    oracle_nn            best training neuron in hindsight
    lookup_{SF,ALL,CSNscatter}   nearest training neuron by pseudobulk PCs / scatter PCs
    ridge{10,1000}_PB_{SF,ALL}   neuron-level ridge PSI ~ pseudobulk PCs
    lowrank_neuron_{PB_SF,PB_ALL,CSNscatter}  PSI = sigmoid(b_e + u_e . f(x_neuron)), binomial loss, neuron rows
    lowrank_cell_{SF,ALL}        same model on single-cell rows (neuron-balanced), held-out neuron = mean over its cells
Outputs -> <out>/human_gen/worm_expr/
"""
import os, glob
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.io import mmread
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from experiments.common.paths import DATA, out_dir

D = str(DATA) + "/"
W = D + "c_elegans/"
OUT = str(out_dir("human_gen", "worm_expr")) + "/"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", 0)); rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
FOLDS = [int(f) for f in os.environ.get("FOLDS", "-1,0,1,2,3,4").split(",")]
PARTS = os.environ.get("PARTS", "1,2").split(",")
MINN = 20                                                      # pooled reads for a (neuron, event) PSI value

# ---------------- PSI ----------------
e = pd.read_csv(W + "events.csv", usecols=["event_id", "event_type", "neuron", "replicate", "nb_reads", "PSI"])
e = e[(e.nb_reads > 0) & e.PSI.notna()].copy(); e["inc"] = e.PSI * e.nb_reads
NEU = sorted(e.neuron.unique()); EVA = np.array(sorted(e.event_id.unique()))
ETYPE = e.drop_duplicates("event_id").set_index("event_id").event_type.reindex(EVA).values
def counts(df):
    g = df.groupby(["neuron", "event_id"])[["inc", "nb_reads"]].sum()
    K = g.inc.unstack().reindex(index=NEU, columns=EVA).fillna(0).values.astype(np.float32)
    N = g.nb_reads.unstack().reindex(index=NEU, columns=EVA).fillna(0).values.astype(np.float32)
    return K, N
def psi(K, N): return np.where(N >= MINN, K / np.maximum(N, 1), np.nan)
K, N = counts(e); P = psi(K, N)
reps = e.groupby("neuron").replicate.unique()
def half_counts(seed):
    r = np.random.default_rng(seed); A = set()
    for n_ in NEU:
        rr = r.permutation(sorted(reps[n_])); A |= set(rr[: (len(rr) + 1) // 2])
    m = e.replicate.isin(A)
    return counts(e[m]), counts(e[~m])
print(f"events {len(EVA)} ({pd.Series(ETYPE).value_counts().to_dict()}), neurons {len(NEU)}, replicates {e.replicate.nunique()}", flush=True)

def within(T, Pm, rows, cols, sd=None, th=0.0, minc=5):
    """within-event Spearman: T, Pm arrays [neuron, event]; rows = neuron idx, cols = bool event mask"""
    t = T[np.ix_(rows, np.where(cols)[0])]; p = Pm[np.ix_(rows, np.where(cols)[0])]
    ok = ~np.isnan(t) & ~np.isnan(p)
    keep = ok.sum(0) >= minc
    if sd is not None: keep &= np.nan_to_num(sd[cols]) >= th
    t, p, ok = t[:, keep], p[:, keep], ok[:, keep]
    if keep.sum() < 5: return np.nan, int(keep.sum())
    tm = np.nansum(np.where(ok, t, 0), 0) / ok.sum(0); pm = np.nansum(np.where(ok, p, 0), 0) / ok.sum(0)
    a = (t - tm)[ok]; b = (p - pm)[ok]
    return spearmanr(a, b).correlation, int(keep.sum())

# ---------------- expression ----------------
genes = pd.read_csv(W + "raw/expression/241119b_scCounts_neuronsBulk.rownames", header=None)[0].values
cells = pd.read_csv(W + "raw/expression/241119b_scCounts_neuronsBulk.colnames", header=None)[0]
ctype = cells.str.rsplit("_", n=1).str[0].values
X = mmread(W + "raw/expression/241119b_scCounts_neuronsBulk.mtx").tocsc().T.tocsr().astype(np.float32)   # cells x genes
lib = np.asarray(X.sum(1)).ravel(); X = X.multiply(1e4 / np.maximum(lib, 1)[:, None]).tocsr(); X.data = np.log1p(X.data)
sf_ids = set(pd.read_csv(W + "raw/list_sf.tsv", sep="\t").gene_id)
is_sf = np.isin(genes, list(sf_ids))
det = np.asarray((X > 0).mean(0)).ravel() >= 0.01
mean_ = np.asarray(X.mean(0)).ravel(); var_ = np.asarray(X.multiply(X).mean(0)).ravel() - mean_ ** 2
hvg = np.zeros(len(genes), bool); cand = np.where(det)[0]; hvg[cand[np.argsort(-var_[cand])[:3000]]] = True
GSETS = {"SF": np.where(is_sf)[0], "ALL": np.where(hvg | is_sf)[0]}
XD = {k: X[:, v].toarray() for k, v in GSETS.items()}
print(f"cells {len(cells)}, SF genes {len(GSETS['SF'])}, ALL genes {len(GSETS['ALL'])}", flush=True)
cell_n = pd.Series(ctype).value_counts().reindex(NEU).values
SC = np.stack([torch.load(W + f"processed/expression/scatter_243/scatter_coeffs_{n_}.pt", map_location="cpu").detach().flatten().numpy()
               for n_ in NEU]).astype(np.float64)

def cos_stats(A):
    An = A / np.linalg.norm(A, axis=1, keepdims=True); C = An @ An.T; o = C[~np.eye(len(A), dtype=bool)]
    return f"mean {o.mean():.4f}  min {o.min():.4f}  max {o.max():.4f}"
def pcs(Xd, fit_rows, k=50):
    mu, sd = Xd[fit_rows].mean(0), Xd[fit_rows].std(0) + 1e-3; Z = (Xd - mu) / sd
    pca = PCA(k, random_state=0).fit(Z[fit_rows]); Z = pca.transform(Z)
    return (Z - Z[fit_rows].mean(0)) / Z[fit_rows].std(0)

# ================= PART 1 =================
if "1" in PARTS:
    L = []
    L.append("== a. CSN scatter-tensor input similarity (cosine between units) ==")
    L.append(f"worm 50 neurons   raw:     {cos_stats(SC)}")
    L.append(f"worm 50 neurons   centred: {cos_stats(SC - SC.mean(0))}")
    hf = sorted(glob.glob(D + "human/processed/expression/scatter_493/scatter_coeffs_*.pt"))
    if hf:
        H = np.stack([torch.load(f, map_location="cpu").detach().flatten().numpy() for f in hf]).astype(np.float64)
        L.append(f"human {len(hf)} tissues raw:     {cos_stats(H)}")
        L.append(f"human {len(hf)} tissues centred: {cos_stats(H - H.mean(0))}")
    PBraw = {k: np.stack([v[ctype == n_].mean(0) for n_ in NEU]) for k, v in XD.items()}
    for k, v in PBraw.items():
        L.append(f"worm pseudobulk log-expr {k}: raw {cos_stats(v)} | centred {cos_stats(v - v.mean(0))}")

    L.append("\n== b. cell -> neuron classification (5-fold CV over cells, 50 classes, chance 0.02; balanced accuracy) ==")
    allc = np.ones(len(cells), bool); y = ctype
    for k in ("SF", "ALL"):
        Z = pcs(XD[k], allc); accs = {"knn15": [], "logreg": []}; pred_all = np.empty(len(y), object)
        for trn, tst in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y):
            p1 = KNeighborsClassifier(15).fit(Z[trn], y[trn]).predict(Z[tst]); accs["knn15"].append(balanced_accuracy_score(y[tst], p1))
            p2 = LogisticRegression(max_iter=3000, C=1.0).fit(Z[trn], y[trn]).predict(Z[tst]); accs["logreg"].append(balanced_accuracy_score(y[tst], p2))
            pred_all[tst] = p2
        rec = pd.Series(pred_all == y).groupby(y).mean()
        L.append(f"{k:3s}: knn15 {np.mean(accs['knn15']):.3f}  logreg {np.mean(accs['logreg']):.3f}   per-neuron recall (logreg) "
                 f"min {rec.min():.2f} ({rec.idxmin()}) median {rec.median():.2f}")
        if k == "SF":
            cf = pd.crosstab(y, pred_all); np.fill_diagonal(cf.values, 0); s = cf.stack().sort_values(ascending=False).head(6)
            L.append("   most confused (true -> pred, cells): " + ", ".join(f"{a}->{b} {c}" for (a, b), c in s.items()))

    L.append("\n== c. split-half pseudobulk identification (half A of a neuron's cells -> nearest half B among 50; 10 repeats) ==")
    for ncell in (None, 20, 5):
        for k in ("SF", "ALL"):
            Xk = XD[k]; acc = []
            for rep in range(10):
                r = np.random.default_rng(rep); A, B = [], []
                for n_ in NEU:
                    idx = r.permutation(np.where(ctype == n_)[0]); h = len(idx) // 2
                    a, b = idx[:h], idx[h:2 * h]
                    if ncell: a, b = a[:ncell], b[:ncell]
                    A.append(Xk[a].mean(0)); B.append(Xk[b].mean(0))
                A, B = np.array(A), np.array(B); mu = (A.mean(0) + B.mean(0)) / 2; sd = np.concatenate([A, B]).std(0) + 1e-3
                A, B = (A - mu) / sd, (B - mu) / sd
                d = ((A[:, None] - B[None]) ** 2).sum(-1); acc.append((d.argmin(1) == np.arange(len(NEU))).mean())
            L.append(f"cells per half {'all' if ncell is None else ncell:>3}  {k:3s}: top-1 {np.mean(acc):.3f}")

    L.append("\n== d. PSI side: neuron identity and split-half within-event ceiling (replicate halves, 10 repeats) ==")
    sd_all = np.nanstd(P, 0)
    ida, ceil = [], {t: [] for t in ("all", "SE", "CE", "CI", "sd>=0.10", "sd>=0.20")}
    for rep in range(10):
        (KA, NA), (KB, NB) = half_counts(rep); PA, PB_ = psi(KA, NA), psi(KB, NB)
        da = PA - np.nanmean(P, 0); db = PB_ - np.nanmean(P, 0)
        S = np.full((len(NEU), len(NEU)), np.nan)
        for i in range(len(NEU)):
            for j in range(len(NEU)):
                ok = ~np.isnan(da[i]) & ~np.isnan(db[j])
                if ok.sum() >= 30: S[i, j] = np.corrcoef(da[i, ok], db[j, ok])[0, 1]
        okr = ~np.isnan(np.diag(S)); ida.append((np.nanargmax(S[okr], 1) == np.where(okr)[0]).mean())
        rows = np.arange(len(NEU))
        for t in ("all", "SE", "CE", "CI"):
            ceil[t].append(within(PA, PB_, rows, np.ones(len(EVA), bool) if t == "all" else ETYPE == t)[0])
        for th in (.1, .2): ceil[f"sd>={th:.2f}"].append(within(PA, PB_, rows, np.ones(len(EVA), bool), sd_all, th)[0])
    L.append(f"replicate-half PSI profile -> own neuron among 50: top-1 {np.mean(ida):.3f} (chance 0.02)")
    L.append("split-half within-event Spearman over 50 neurons: " + "  ".join(f"{k} {np.nanmean(v):.3f}" for k, v in ceil.items()))

    L.append("\n== e. Mantel: neuron-pair expression distance vs PSI-profile distance (Spearman over pairs) ==")
    dev_ = P - np.nanmean(P, 0); Dp = np.full((len(NEU), len(NEU)), np.nan)
    for i in range(len(NEU)):
        for j in range(i + 1, len(NEU)):
            ok = ~np.isnan(dev_[i]) & ~np.isnan(dev_[j])
            if ok.sum() >= 30: Dp[i, j] = 1 - np.corrcoef(dev_[i, ok], dev_[j, ok])[0, 1]
    iu = np.triu_indices(len(NEU), 1); okp = ~np.isnan(Dp[iu])
    for k, v in list(PBraw.items()) + [("CSNscatter", SC)]:
        Zc = (v - v.mean(0)) / (v.std(0) + 1e-6); De = ((Zc[:, None] - Zc[None]) ** 2).sum(-1) ** .5
        L.append(f"{k:10s}: rho {spearmanr(De[iu][okp], Dp[iu][okp]).correlation:.3f}  (pairs {okp.sum()})")
    txt = "\n".join(L); open(OUT + "part1.txt", "w").write(txt + "\n"); print(txt, flush=True)

# ================= PART 2 =================
class LowRank(nn.Module):
    def __init__(self, d, b0, k=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, k))
        self.u = nn.Parameter(0.01 * torch.randn(len(b0), k)); self.b = nn.Parameter(torch.tensor(b0, dtype=torch.float32))
    def forward(self, x): return self.f(x) @ self.u.T + self.b
def binom(lg, k, n):
    w = n.sqrt() / (n + 1e-6)
    return -((k * nn.functional.logsigmoid(lg) + (n - k) * nn.functional.logsigmoid(-lg)) * w).sum() / ((n * w).sum() + 1e-6)

results = []
if "2" in PARTS:
    perm = list(np.random.default_rng(123).permutation(NEU))
    sp = {k: sorted(pd.read_csv(W + f"splits/cell/{k}.csv", usecols=["neuron"]).neuron.unique()) for k in ("train", "valid", "test")}
    for FOLD in FOLDS:
        if FOLD < 0: TR, VT, TE = sp["train"], sp["valid"], sp["test"]
        else:
            TE = sorted(perm[FOLD::5]); rest = [n_ for n_ in perm if n_ not in TE]; VT = sorted(rest[:5]); TR = sorted(rest[5:])
        it, iv, ie = ([NEU.index(n_) for n_ in s] for s in (TR, VT, TE))
        evm = ((N[it] >= MINN).sum(0) >= 5)                              # events chosen on training neurons only
        sd_tr = np.nanstd(P[it], 0)
        print(f"\n=== fold {FOLD}: test {TE}\n    valid {VT}; events {evm.sum()} ({pd.Series(ETYPE[evm]).value_counts().to_dict()})", flush=True)
        Ptr = np.nanmean(np.where(N[it] >= MINN, K[it] / np.maximum(N[it], 1), np.nan), 0)
        def score(name, Pred, extra=None):
            for role, rows in (("valid", iv), ("test", ie)):
                row = dict(method=name, fold=FOLD, split=role)
                for t in ("all", "SE", "CE", "CI"):
                    cm = evm & (np.ones(len(EVA), bool) if t == "all" else ETYPE == t)
                    row[f"within_{t}"], row[f"events_{t}"] = within(P, Pred, rows, cm)
                for th in (.1, .2):
                    row[f"within_sd>={th:.2f}"], row[f"events_sd>={th:.2f}"] = within(P, Pred, rows, evm, sd_tr, th)
                if extra: row.update(extra)
                results.append(row)
                if role == "test": print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if not k.startswith("events")}, flush=True)
        def vscore(Pred): return within(P, Pred, iv, evm)[0]
        # ceiling: replicate split-half on the held-out neurons (5 repeats)
        rows_ = []
        for rep in range(5):
            (KA, NA), (KB, NB) = half_counts(rep); PA, PB_ = psi(KA, NA), psi(KB, NB)
            r_ = {}
            for t in ("all", "SE", "CE", "CI"):
                r_[f"within_{t}"] = within(PA, PB_, ie, evm & (np.ones(len(EVA), bool) if t == "all" else ETYPE == t))[0]
            for th in (.1, .2): r_[f"within_sd>={th:.2f}"] = within(PA, PB_, ie, evm, sd_tr, th)[0]
            rows_.append(r_)
        results.append(dict(method="ceiling_splithalf", fold=FOLD, split="test", **pd.DataFrame(rows_).mean().to_dict()))
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in results[-1].items()}, flush=True)
        # random neuron, oracle nn
        rr = []
        for d_ in range(20):
            Pred = np.full_like(P, np.nan); pick = np.random.default_rng(d_).choice(it, len(NEU)); Pred[:] = P[pick]
            rr.append(Pred)
        for role, rows in (("valid", iv), ("test", ie)):
            row = dict(method="random_neuron", fold=FOLD, split=role)
            for t in ("all", "SE", "CE", "CI"):
                cm = evm & (np.ones(len(EVA), bool) if t == "all" else ETYPE == t)
                row[f"within_{t}"] = np.nanmean([within(P, Pr, rows, cm)[0] for Pr in rr])
            for th in (.1, .2): row[f"within_sd>={th:.2f}"] = np.nanmean([within(P, Pr, rows, evm, sd_tr, th)[0] for Pr in rr])
            results.append(row)
        dtr = P[it] - Ptr
        Pred = np.full_like(P, np.nan)
        for i in iv + ie:
            di = P[i] - Ptr; best = (-2, None)
            for j, s_ in enumerate(it):
                ok = evm & ~np.isnan(di) & ~np.isnan(dtr[j])
                if ok.sum() >= 10:
                    c = spearmanr(di[ok], dtr[j, ok]).correlation
                    if c > best[0]: best = (c, s_)
            if best[1] is not None: Pred[i] = P[best[1]]
        score("oracle_nn", Pred)
        # inputs
        trc = np.isin(ctype, TR)
        FE = {k: pcs(XD[k], trc) for k in ("SF", "ALL")}
        PBf = {k: np.stack([FE[k][ctype == n_].mean(0) for n_ in NEU]) for k in FE}
        sc = (SC - SC[it].mean(0)) / (SC[it].std(0) + 1e-6)
        pca = PCA(min(20, len(it) - 1), random_state=0).fit(sc[it]); Zsc = pca.transform(sc); Zsc = (Zsc - Zsc[it].mean(0)) / (Zsc[it].std(0) + 1e-6)
        INP = {"PB_SF": PBf["SF"], "PB_ALL": PBf["ALL"], "CSNscatter": Zsc}
        # lookup
        for nm, Z in (("SF", INP["PB_SF"]), ("ALL", INP["PB_ALL"]), ("CSNscatter", Zsc)):
            Pred = np.full_like(P, np.nan)
            for i in iv + ie: Pred[i] = P[it[int(np.argmin(((Z[it] - Z[i]) ** 2).sum(1)))]]
            score(f"lookup_{nm}", Pred)
        # ridge on neuron pseudobulk
        Ytr = np.where(np.isnan(P[it]), Ptr, P[it]); Ytr = np.nan_to_num(Ytr, nan=0.5); yb = Ytr.mean(0)
        for nm in ("PB_SF", "PB_ALL"):
            for lam in (10.0, 1000.0):
                A = INP[nm][it]; Wr = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr - yb))
                score(f"ridge{int(lam)}_{nm}", np.clip(INP[nm] @ Wr + yb, 0, 1))
        # low-rank, neuron rows
        EVi = np.where(evm)[0]; m0 = np.clip(np.nan_to_num(Ptr[EVi], nan=0.5), .02, .98); b0 = np.log(m0 / (1 - m0))
        Kt = torch.tensor(K[it][:, EVi], device=dev); Nt = torch.tensor(N[it][:, EVi], device=dev)
        def full(Psub):
            out = np.full_like(P, np.nan); out[:, EVi] = Psub; return out
        for nm, Xin in INP.items():
            torch.manual_seed(SEED); Xt = torch.tensor(Xin, dtype=torch.float32, device=dev)
            model = LowRank(Xt.shape[1], b0).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best, bs = -2, None
            for step in range(3000):
                model.train(); loss = binom(model(Xt[it]), Kt, Nt); opt.zero_grad(); loss.backward(); opt.step()
                if step % 100 == 99:
                    model.eval()
                    with torch.no_grad(): s = vscore(full(torch.sigmoid(model(Xt)).cpu().numpy()))
                    if s > best: best, bs = s, {k_: v.clone() for k_, v in model.state_dict().items()}
            model.load_state_dict(bs); model.eval()
            with torch.no_grad(): score(f"lowrank_neuron_{nm}", full(torch.sigmoid(model(Xt)).cpu().numpy()), dict(best_valid=best))
        # low-rank, cell rows (neuron-balanced sampling); prediction for a neuron = mean over its cells
        cidx = np.where(trc)[0]; cpos = np.array([it.index(NEU.index(c)) for c in ctype[cidx]])
        pw = 1.0 / pd.Series(cpos).map(pd.Series(cpos).value_counts()).values; pw /= pw.sum()
        for nm in ("SF", "ALL"):
            torch.manual_seed(SEED); Zc = torch.tensor(FE[nm], dtype=torch.float32, device=dev)
            model = LowRank(Zc.shape[1], b0).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            def neuron_pred():
                model.eval(); out = np.full((len(NEU), len(EVi)), np.nan, np.float32)
                with torch.no_grad():
                    for i in range(len(NEU)):
                        out[i] = torch.sigmoid(model(Zc[torch.tensor(np.where(ctype == NEU[i])[0], device=dev)])).mean(0).cpu().numpy()
                return full(out)
            best, bs = -2, None
            for step in range(4000):
                model.train(); b = rng.choice(len(cidx), 512, p=pw)
                cb = torch.tensor(cpos[b], device=dev); loss = binom(model(Zc[torch.tensor(cidx[b], device=dev)]), Kt[cb], Nt[cb])
                opt.zero_grad(); loss.backward(); opt.step()
                if step % 200 == 199:
                    s = vscore(neuron_pred())
                    if s > best: best, bs = s, {k_: v.clone() for k_, v in model.state_dict().items()}
            model.load_state_dict(bs); score(f"lowrank_cell_{nm}", neuron_pred(), dict(best_valid=best))
        pd.DataFrame(results).to_csv(OUT + f"part2_results_seed{SEED}.csv", index=False)

    res = pd.DataFrame(results); res.to_csv(OUT + f"part2_results_seed{SEED}.csv", index=False)
    cols = ["within_all", "within_SE", "within_CE", "within_CI", "within_sd>=0.10", "within_sd>=0.20"]
    t = res[res.split == "test"]; pd.set_option("display.width", 250)
    L = ["== Part 2: held-out neurons, within-event Spearman (test) =="]
    rot = t[t.fold >= 0]
    if len(rot):
        g = rot.groupby("method")[cols]; s = g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)
        L += [f"-- rotation folds {sorted(rot.fold.unique())} (mean ± sd)", s.to_string()]
    fr = t[t.fold < 0]
    if len(fr): L += ["-- frozen 'cell' split (same held-out neurons as the CSN cell split)", fr.set_index("method")[cols].round(3).to_string()]
    txt = "\n".join(L); open(OUT + f"part2_summary_seed{SEED}.txt", "w").write(txt + "\n"); print(txt, flush=True)
