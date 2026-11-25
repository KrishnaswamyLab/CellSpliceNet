"""How much tissue-specific (within-event) signal is there in the human ASCOT/GTEx PSI, and how
much of it is measurement noise?  (Step 0 of the Nov-24 plan.)

ASCOT gives one aggregated PSI per (exon, tissue) and no read counts, so a split-half reliability
is impossible.  Instead:
  1. variance decomposition: within-event share overall and by event variability tier
  2. near-replicate tissue pairs (e.g. Cerebellum vs Cerebellar Hemisphere): correlation of their
     deviations d = PSI - mean over the OTHER tissues.  Two noisy readouts of (almost) the same
     biology -> this correlation is a lower bound on the reliability of a deviation.  Unrelated
     random tissue pairs are the control.  Stratified by event SD (computed on the other tissues,
     so independent of the pair's noise) and by gene coverage (ASCOT gtex_nauc, min of the pair).
  3. defines the tissue-variable event subset (SD over tissues >= 0.05 / 0.10 / 0.20) and writes it
  4. re-scores pilot CSN predictions and the baselines on those subsets (within-event Spearman, same
     definition as pilot_report.py)
Outputs -> <out>/human_gen/ceiling/
"""
import glob, os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

from experiments.human_gen.common import ASCOT_NAUC, PARTS, baselines_file, out, read_split, split_names, split_runs

OUT = str(out("ceiling")) + "/"
rng = np.random.default_rng(0)

allr = pd.concat([read_split("gene", k, usecols=["gene_id", "event_id", "PSI", "neuron"])
                  for k in PARTS], ignore_index=True)
W = allr.pivot_table(index="event_id", columns="neuron", values="PSI")
gene = allr.drop_duplicates("event_id").set_index("event_id").gene_id.str.split(".").str[0]
print(f"events={len(W):,} tissues={W.shape[1]} rows={len(allr):,}", flush=True)

# ---------------- 1. variance decomposition ----------------
n_t = W.notna().sum(1)
mu = W.mean(1); sd = W.std(1)
dev = W.sub(mu, axis=0)
tot = np.nanvar(W.values); within_var = np.nanmean(dev.values ** 2)
lines = [f"total var {tot:.4f}  within-event var {within_var:.4f}  share {within_var / tot:.3%}"]
ok = n_t >= 5
for th in (0.02, 0.05, 0.10, 0.20):
    sel = ok & (sd >= th)
    sh = np.nansum(dev[sel].values ** 2) / np.nansum(dev[ok].values ** 2)
    lines.append(f"SD>={th:.2f}: {sel.sum():,} events ({sel.mean():.1%} of all, {sel.sum() / ok.sum():.1%} of >=5-tissue)"
                 f" carry {sh:.1%} of within-event variance; mean PSI of these {mu[sel].mean():.2f}")
lines.append("event SD quantiles (>=5 tissues): " + " ".join(f"q{int(q * 100)}={sd[ok].quantile(q):.3f}" for q in (.25, .5, .75, .9, .95, .99)))
# mean-variance relation: deviations are largest at intermediate PSI
bins = pd.cut(mu[ok], [0, .05, .2, .5, .8, .95, 1.0001], include_lowest=True)
lines.append("median event SD by mean PSI bin: " + ", ".join(f"{b}: {v:.3f}" for b, v in sd[ok].groupby(bins, observed=True).median().items()))

# ---------------- 2. near-replicate tissue pairs ----------------
nauc = pd.read_csv(ASCOT_NAUC, encoding="utf-8-sig")
nauc["gid"] = nauc.gene_id.str.split(".").str[0]
nauc = nauc.drop_duplicates("gid").set_index("gid")
PAIRS = [("Cerebellum - Brain", "Cerebellar Hemisphere - Brain"),
         ("Cortex - Brain", "Frontal Cortex - Brain"),
         ("Caudate nucleus - Brain", "Putamen - Brain"),
         ("Not Sun Exposed - Skin", "Sun Exposed (Lower leg) - Skin"),
         ("Subcutaneous - Adipose", "Visceral (Omentum) - Adipose"),
         ("Coronary - Artery", "Tibial - Artery"),
         ("Gastroesoph. Junc. - Esophagus", "Muscularis - Esophagus")]
PAIRS = [p for p in PAIRS if p[0] in W.columns and p[1] in W.columns]
tis = list(W.columns)
far = set()  # unrelated control pairs: different tissue families, none from the brain cluster on both sides
while len(far) < 30:
    a, b = rng.choice(tis, 2, replace=False)
    if a.split(" - ")[-1] != b.split(" - ")[-1]: far.add(tuple(sorted((a, b))))

def pair_stats(a, b):
    others = [t for t in tis if t not in (a, b)]
    ref = W[others]; m_o = ref.mean(1); sd_o = ref.std(1); n_o = ref.notna().sum(1)
    d = pd.DataFrame({"da": W[a] - m_o, "db": W[b] - m_o, "sd_o": sd_o, "n_o": n_o})
    d["cov"] = np.minimum(gene.reindex(d.index).map(nauc[a]) if a in nauc else np.nan,
                          gene.reindex(d.index).map(nauc[b]) if b in nauc else np.nan)
    d = d.dropna(subset=["da", "db"]); d = d[d.n_o >= 5]
    out = []
    def add(tag, x):
        if len(x) < 50: return
        out.append(dict(a=a, b=b, stratum=tag, n=len(x), spearman=spearmanr(x.da, x.db).correlation,
                        pearson=pearsonr(x.da, x.db)[0], sd_dev=np.sqrt(((x.da ** 2 + x.db ** 2) / 2).mean())))
    add("all", d)
    for lo, hi in ((0, .05), (.05, .10), (.10, .20), (.20, 1)):
        add(f"sd_other[{lo:.2f},{hi:.2f})", d[(d.sd_o >= lo) & (d.sd_o < hi)])
    if d["cov"].notna().sum() > 300:
        q = d["cov"].quantile([1 / 3, 2 / 3]).values
        for tag, x in (("cov_low", d[d["cov"] <= q[0]]), ("cov_mid", d[(d["cov"] > q[0]) & (d["cov"] <= q[1])]),
                       ("cov_high", d[d["cov"] > q[1]])):
            add(tag, x)
            add(tag + "&sd>=0.10", x[x.sd_o >= .10])
    return out

rows = []
for a, b in PAIRS: rows += [dict(kind="near_replicate", **r) for r in pair_stats(a, b)]
for a, b in sorted(far): rows += [dict(kind="unrelated", **r) for r in pair_stats(a, b)]
pr = pd.DataFrame(rows); pr.to_csv(OUT + "tissue_pair_reliability.csv", index=False)
summ = pr.groupby(["kind", "stratum"])[["n", "spearman", "pearson", "sd_dev"]].mean().round(3)

# ---------------- 3. variable-event subset ----------------
# SD over all tissues (a property of the target, used only to stratify evaluation) and over the
# 34 training tissues of splits/tissue (usable for training-time selection without test leakage)
tr_tis = sorted(read_split("gene_x_cell", "train", usecols=["neuron"]).neuron.unique())
ev = pd.DataFrame({"n_tissues": n_t, "mean": mu, "sd_all": sd, "range_all": W.max(1) - W.min(1),
                   "sd_train_tissues": W[tr_tis].std(1)})
ev.to_csv(OUT + "event_variability.csv")

# ---------------- 4. re-score pilot predictions + baselines on variable subsets ----------------
def within(d, c):
    d = d[d.groupby("event_id").neuron.transform("count") >= 5]
    if d.empty: return np.nan
    a = d.PSI - d.groupby("event_id").PSI.transform("mean"); b = d[c] - d.groupby("event_id")[c].transform("mean")
    return spearmanr(a, b).correlation if b.std() > 1e-9 else 0.0

TIERS = [("all", 0.0), ("sd>=0.05", .05), ("sd>=0.10", .10), ("sd>=0.20", .20)]
def score(d, c, **kw):
    out = []
    for tag, th in TIERS:
        x = d[d.event_id.map(ev.sd_all).fillna(0) >= th].dropna(subset=[c])
        if x.event_id.nunique() < 20: continue
        out.append(dict(tier=tag, events=x.event_id.nunique(), n=len(x), pooled=spearmanr(x.PSI, x[c]).correlation,
                        within=within(x, c), **kw))
    return out

res = []
BASE = {"gene_k8": (baselines_file("gene_k8"), ["tissue_offset", "gbm_plus_off"]),
        "chrom_k8": (baselines_file("chrom_k8"), ["tissue_offset", "gbm_plus_off"]),
        "cell_k8": (baselines_file("cell"), ["nn_tissue", "knn3_tissue"])}
for split, (bf, bcols) in BASE.items():
    base = pd.read_csv(bf)
    evaluated = None
    names = "|".join(split_names(split))
    for run in split_runs("hg", split):
        m = re.search(rf"hg_(?:{names})_(.+)_s(\d+)$", run); model, seed = m.group(1), int(m.group(2))
        if "nocues" in model: continue
        f = f"{run}/preds_best_validation_model.csv"
        if not os.path.exists(f): continue
        p = pd.read_csv(f).rename(columns={"psi_target": "PSI", "psi_pred": "csn"})
        evaluated = p[["event_id", "neuron"]].drop_duplicates()
        if split == "cell_k8":
            p = p.merge(base[["event_id", "neuron", "gene_seen"]], on=["event_id", "neuron"], how="left")
            p = p[p.gene_seen == 1]
        cols = [("csn", "")] + ([("dev_pred", "[dev]")] if "dev_pred" in p.columns else [])
        for c, sfx in cols: res += score(p, c, split=split, model=model + sfx, seed=seed)
    if evaluated is not None:
        b = base.merge(evaluated, on=["event_id", "neuron"])
        if split == "cell_k8": b = b[b.gene_seen == 1]
        for c in bcols: res += score(b, c, split=split, model="BASELINE:" + c, seed=0)
rs = pd.DataFrame(res); rs.to_csv(OUT + "within_by_variability.csv", index=False)
rsum = rs.groupby(["split", "model", "tier"])[["events", "pooled", "within"]].mean().round(3).unstack("tier")

with open(OUT + "summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n\n== deviation correlation, tissue pairs (mean over pairs) ==\n")
    f.write(summ.to_string() + "\n\n== per near-replicate pair, all events ==\n")
    f.write(pr[(pr.kind == "near_replicate") & (pr.stratum == "all")].round(3).to_string() + "\n\n")
    f.write("== within-event Spearman by event-variability tier (best-valid ckpt; cell_k8 = seen genes) ==\n")
    f.write(rsum["within"].to_string() + "\n\n== events per tier ==\n" + rsum["events"].to_string() + "\n")
print(open(OUT + "summary.txt").read(), flush=True)
