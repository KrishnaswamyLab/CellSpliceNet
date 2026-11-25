"""Data checks for the donor-level / expression-only tissue generalization plan (Nov 24).

A. alignment: per tissue, sample IDs in geneCounts vs k_j_counts vs sampleAnnotation; donors overall
B. exon-specificity of tissue effects (ASCOT matrix): share of within-event variance explained by a per-tissue
   offset; rank of the exon x tissue deviation matrix (SVD variance explained) -> how big k must be in u_e . v_t
C. donor-level signal in 6 tissues (per-sample junction PSI):
   - across-donor PSI variance vs binomial read-noise expectation (overdispersion ratio)
   - 5-fold CV ridge of per-sample PSI on expression PCs: splicing factors (493 SF) vs all-gene PCs vs
     shuffled-donor control -> is there an expression -> splicing signal to learn at donor level?
Outputs -> <out>/human_gen/donor_checks/
"""
import os, sys, zlib
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from experiments.human_gen.common import EVENT_VARIABILITY, HUMAN, SF_ORDER, all_rows, out
from processing.human.junctions import counts_dir, junction_keys, tissue_folders

EC = str(counts_dir(HUMAN / "raw")) + "/"
OUT = str(out("donor_checks")) + "/"
_, J = junction_keys(HUMAN / "raw", all_rows(["event_id"]).event_id.unique())   # junction keys of every exon
need = set(pd.concat(J.values()))
log = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)

# ---------------- A. alignment ----------------
folders = tissue_folders(HUMAN / "raw")
rows, donors = [], set()
for f in folders:
    g = pd.read_csv(EC + f + "/geneCounts.tsv.gz", sep="\t", nrows=1).columns[1:]
    k = pd.read_csv(EC + f + "/k_j_counts.tsv.gz", sep="\t", nrows=1).columns[4:]
    a = pd.read_csv(EC + f + "/sampleAnnotation.tsv", sep="\t")
    donors |= set(a.INDIVIDUAL_ID)
    rows.append(dict(tissue=f.split("--")[0], gene=len(g), junc=len(k), annot=len(a), gene_and_junc=len(set(g) & set(k)),
                     donors=a.INDIVIDUAL_ID.nunique()))
A = pd.DataFrame(rows)
p("== A. sample alignment ==\n" + A.to_string(index=False))
p(f"total samples (gene&junc) {A.gene_and_junc.sum():,}; distinct donors {len(donors):,}; "
  f"tissues with gene==junc==annot: {(A.gene == A.gene_and_junc).sum()}/{len(A)}")

# ---------------- B. exon-specificity of tissue effects ----------------
allr = all_rows(["event_id", "PSI", "neuron"])
W = allr.pivot_table(index="event_id", columns="neuron", values="PSI")
W = W[W.notna().sum(1) >= 40]
Dv = W.sub(W.mean(1), axis=0)
ev = pd.read_csv(EVENT_VARIABILITY, index_col=0)
p("\n== B. tissue effects: offset vs exon-specific ==")
for tag, sel in (("all", Dv.index), ("sd>=0.05", Dv.index[ev.sd_all.reindex(Dv.index) >= .05]),
                 ("sd>=0.10", Dv.index[ev.sd_all.reindex(Dv.index) >= .10])):
    d = Dv.loc[sel]; ss = np.nansum(d.values ** 2); off = d.mean(0)
    r2_off = 1 - np.nansum((d - off).values ** 2) / ss
    X = d.fillna(0).values; s = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2; c = np.cumsum(s) / s.sum()
    p(f"{tag}: events={len(d):,} tissue-offset R2={r2_off:.3f} | SVD cum. var k=1:{c[0]:.2f} 2:{c[1]:.2f} 4:{c[3]:.2f} "
      f"8:{c[7]:.2f} 16:{c[15]:.2f}")

# ---------------- C. donor-level signal ----------------
SF = [l.strip() for l in open(SF_ORDER) if l.strip()]
TIS = ["Liver", "Muscle_-_Skeletal", "Brain_-_Cortex", "Whole_Blood", "Lung", "Thyroid"]
p("\n== C. donor-level signal (exons with >=20 reads in >=90% of samples) ==")
crow = []
for t in TIS:
    f = [x for x in folders if x.startswith(t + "--")][0]
    parts = []
    for ch in pd.read_csv(EC + f + "/k_j_counts.tsv.gz", sep="\t", chunksize=100_000):
        key = ch.seqnames.astype(str) + ":" + ch.start.astype(str) + ":" + ch.end.astype(str)
        m = key.isin(need)
        if m.any(): x = ch.loc[m].iloc[:, 4:]; x.index = key[m]; parts.append(x)
    K = pd.concat(parts).groupby(level=0).sum()
    G = pd.read_csv(EC + f + "/geneCounts.tsv.gz", sep="\t", index_col=0)
    samples = [s for s in K.columns if s in G.columns]
    K = K[samples]; G = G[samples]
    inc = (K.reindex(J["inc1"].values).fillna(0).values + K.reindex(J["inc2"].values).fillna(0).values) / 2
    exc = K.reindex(J["skip"].values).fillna(0).values
    tot = inc + exc
    keep = (tot >= 20).mean(1) >= 0.9
    psi = np.where(tot >= 20, inc / np.maximum(tot, 1e-9), np.nan)[keep]
    tt = tot[keep]
    pm = np.nanmean(psi, 1, keepdims=True)
    psi_f = np.where(np.isnan(psi), pm, psi)
    var_obs = np.nanvar(psi, 1)
    var_bin = np.nanmean(np.where(tt >= 20, pm * (1 - pm) / np.maximum(tt, 1), np.nan), 1)
    ratio = var_obs / np.maximum(var_bin, 1e-12)
    sdd = np.sqrt(var_obs)
    sel = (sdd >= 0.03) & (ratio >= 2)       # exons with real donor-to-donor variation
    # expression features
    G.index = G.index.str.split(".").str[0]; G = G[~G.index.duplicated()]
    L = np.log1p(G / G.sum(0) * 1e6)
    Xsf = PCA(20, random_state=0).fit_transform(((L.reindex(SF).dropna().T) - L.reindex(SF).dropna().T.mean()).values)
    Lall = L[L.mean(1) > 1]
    Xall = PCA(20, random_state=0).fit_transform((Lall.T - Lall.T.mean()).values)
    Y = psi_f[sel].T                                              # samples x exons
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    kf = KFold(5, shuffle=True, random_state=0)
    def cvr2(X, Yy):
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        P = cross_val_predict(Ridge(alpha=10.0), X, Yy, cv=kf)
        return 1 - ((Yy - P) ** 2).sum(0) / (Yy ** 2).sum(0)
    rng = np.random.default_rng(zlib.crc32(t.encode()))
    r_sf = cvr2(Xsf, Y); r_all = cvr2(Xall, Y); r_sh = cvr2(Xsf[rng.permutation(len(Xsf))], Y)
    row = dict(tissue=t, samples=len(samples), exons_covered=int(keep.sum()),
               overdisp_median=float(np.median(ratio)), frac_ratio_gt2=float((ratio >= 2).mean()),
               exons_donor_var=int(sel.sum()), median_donor_sd=float(np.median(sdd[sel])) if sel.any() else np.nan,
               cvR2_SF_median=float(np.median(r_sf)), cvR2_SF_frac_gt0_1=float((r_sf > .1).mean()),
               cvR2_ALL_median=float(np.median(r_all)), cvR2_shuffled_median=float(np.median(r_sh)))
    crow.append(row); p(row)
C = pd.DataFrame(crow); C.to_csv(OUT + "donor_signal.csv", index=False)
p("\n" + C.round(3).to_string(index=False))
open(OUT + "summary.txt", "w").write("\n".join(log) + "\n")
