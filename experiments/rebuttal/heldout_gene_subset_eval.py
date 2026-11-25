import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
"""Held-out-gene / held-out-event test accuracy of the grouped full models by gene subset (expression breadth,
HVG). Needs gene_selection_characterize.py output and dump_preds.py predictions of <runs>/fullmodel_grouped."""
from scipy.stats import spearmanr
from experiments.common.paths import out_dir, read_split
from experiments.worm_ablations.common import EVENTS_COORDS, GENE_IDS, GROUPED, SC_EXPR, read_preds, run_preds
g=pd.read_csv(out_dir("rebuttal")/"gene_char.csv",index_col=0)
hvg=set(g[g.hvg2000].index); restr=set(g[g.breadth<=0.2].index); broad=set(g[g.breadth>=0.5].index)
def wr(d,c):
    d=d[d.groupby("event_id").neuron.transform("count")>=5]
    if d.event_id.nunique()<5: return np.nan
    a=d.target-d.groupby("event_id").target.transform("mean"); b=d[c]-d.groupby("event_id")[c].transform("mean")
    return spearmanr(a,b).correlation if b.std()>1e-9 else 0.0
for s in ["genebased","eventbased"]:
    te=read_split(GROUPED[s],"test","c_elegans")[["event_id","neuron","gene_id"]]
    d=None
    for k in (1,2,3):
        p=read_preds(run_preds("fullmodel_grouped",s,k))[["event_id","neuron","target","pred"]].rename(columns={"pred":f"m{k}"})
        d=p if d is None else d.merge(p.drop(columns="target"),on=["event_id","neuron"])
    d=d.merge(te,on=["event_id","neuron"])
    sd=d.groupby("event_id").target.std()
    hiv=set(sd[sd>=sd.median()].index)
    subsets=[("all test genes",d),("HVG top-2000 genes (expression)",d[d.gene_id.isin(hvg)]),
             ("neuron-restricted genes (<=20% types)",d[d.gene_id.isin(restr)]),
             ("broadly expressed genes (>=50% types)",d[d.gene_id.isin(broad)]),
             ("high splicing-variability events*",d[d.event_id.isin(hiv)])]
    print(f"\n== {s} held-out test (our full model, 3 seeds: mean [min,max]) ==")
    print(f"  {'subset':40s} genes events rows   pooled rho              within-event rho")
    for lab,x in subsets:
        pr=[spearmanr(x.target,x[f"m{k}"]).correlation for k in (1,2,3)]; wv=[wr(x,f"m{k}") for k in (1,2,3)]
        print(f"  {lab:40s} {x.gene_id.nunique():5d} {x.event_id.nunique():6d} {len(x):5d}   "
              f"{np.mean(pr):.3f} [{min(pr):.2f},{max(pr):.2f}]    {np.nanmean(wv):.3f} [{np.nanmin(wv):.2f},{np.nanmax(wv):.2f}]")
print("\n* defined from the test PSI itself (above-median across-neuron SD) — selects on the outcome")
