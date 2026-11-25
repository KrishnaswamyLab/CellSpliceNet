"""Score AlphaGenome native-splicing PSI vs CellSpliceNet on identical GTEx test rows."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, pearsonr
import glob, os, sys
from experiments.common.paths import out_dir
# usage: compare_ag_csn.py [CSN predictions csv ...]  (default: every preds_*.csv of
# experiments/human_full/eval_gtex_subsample.py in <out>/human_full/subsample_eval)
ag = pd.read_csv(out_dir("comparisons", "alphagenome") / "ag_gtex_psi.csv")
seeds = {}
for f in sys.argv[1:] or sorted(glob.glob(str(out_dir("human_full", "subsample_eval") / "preds_*.csv"))):
    tag = os.path.basename(f)[len("preds_"):-len(".csv")]
    p = pd.read_csv(f)
    seeds[tag] = p[["event_id", "neuron", "psi_target", "psi_pred"]].rename(columns={"psi_pred": f"csn_{tag}"})
def within(d, c):
    d = d[d.groupby("event_id").neuron.transform("count") >= 5]
    a = d.target - d.groupby("event_id").target.transform("mean"); b = d[c] - d.groupby("event_id")[c].transform("mean")
    return spearmanr(a, b).correlation
print(f"AlphaGenome: {len(ag)} rows / {ag.event_id.nunique()} events; PSI undefined (no junction predicted): {ag.ag_psi.isna().mean():.1%}")
res = []
for tag, p in seeds.items():
    d = p.merge(ag, on=["event_id", "neuron"]).dropna(subset=["ag_psi"]).rename(columns={"psi_target": "target"})
    c = f"csn_{tag}"
    r = dict(seed=tag, rows=len(d), events=d.event_id.nunique(),
             csn_spearman=spearmanr(d.target, d[c]).correlation, ag_spearman=spearmanr(d.target, d.ag_psi).correlation,
             csn_pearson=pearsonr(d.target, d[c])[0], ag_pearson=pearsonr(d.target, d.ag_psi)[0],
             csn_within=within(d, c), ag_within=within(d, "ag_psi"))
    res.append(r)
    if tag == "original":   # event-clustered bootstrap on one seed
        g = {e: x for e, x in d.groupby("event_id")}; ev = np.array(list(g)); rng = np.random.default_rng(0); bs = []
        for _ in range(500):
            s = pd.concat([g[e] for e in rng.choice(ev, len(ev))])
            bs.append((spearmanr(s.target, s[c]).correlation, spearmanr(s.target, s.ag_psi).correlation))
        bs = np.array(bs)
        ci = {k: np.percentile(v, [2.5, 97.5]).round(3) for k, v in [("csn", bs[:, 0]), ("ag", bs[:, 1]), ("ag-csn", bs[:, 1] - bs[:, 0])]}
r = pd.DataFrame(res)
pd.set_option("display.width", 200); print(r.round(3).to_string(index=False))
print("\nmean +/- SD over 3 CSN seeds:")
for k in ["spearman", "pearson", "within"]:
    print(f"  {k:9s} CSN {r[f'csn_{k}'].mean():.3f} +/- {r[f'csn_{k}'].std():.3f}   AlphaGenome {r[f'ag_{k}'].mean():.3f}")
print("bootstrap 95% CI (original seed, event-clustered):", ci)
