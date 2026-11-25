import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
"""Released checkpoint vs lookup baselines on the row-level SE and grouped test sets, split by whether the test
event was in the released model's training data (predictions from eval_released.sh)."""
from sklearn.ensemble import GradientBoostingRegressor

from experiments.common.paths import out_dir, read_split
from experiments.worm_ablations.common import EVENTS_COORDS, GROUPED, read_preds

tr=read_split("rows_se","train","c_elegans")[["event_id","neuron","nb_reads","PSI","gene_id","p_seq_len","sj_seq_len"]].rename(columns={"PSI":"psi"})
co=pd.read_csv(EVENTS_COORDS,sep="\t").drop_duplicates("event_id")
CC=["exon_start","exon_end","intron_start","intron_end","gene_length"]
co=co[["event_id","gene_id"]+CC]
tr=tr.drop(columns=[c for c in ["gene_id"] if c in tr.columns]).merge(co,on="event_id",how="left")
gm=tr.psi.mean(); ev=tr.groupby("event_id").psi.mean(); gn=tr.groupby("gene_id").psi.mean()
tr["r"]=tr.psi-tr.event_id.map(ev); noff=tr.groupby("neuron").r.mean()
def feats(d):
    return pd.DataFrame({"exon_len":d.exon_end-d.exon_start,"up":d.exon_start-d.intron_start,"dn":d.intron_end-d.exon_end,
        "glen":d.gene_length,"rel":d.exon_start/d.gene_length,"m3":((d.exon_end-d.exon_start)%3==0).astype(int)})
tre=tr.groupby("event_id").agg(psi=("psi","mean"),**{c:(c,"first") for c in CC})
gbm=GradientBoostingRegressor(random_state=0,max_depth=3,n_estimators=200,learning_rate=0.05).fit(feats(tre),tre.psi)
def wr(d,c):
    d=d[d.groupby("event_id").neuron.transform("count")>=5]
    if len(d)<10: return np.nan
    a=d.target-d.groupby("event_id").target.transform("mean"); b=d[c]-d.groupby("event_id")[c].transform("mean")
    return spearmanr(a,b).correlation if b.std()>0 else 0.0
def sp(d,c): return spearmanr(d.target,d[c]).correlation if len(d)>2 and d[c].std()>0 else np.nan
allun=[]
for s in ["SE","eventbased","genebased","neuronbased"]:
    d=read_preds(out_dir("rebuttal","released")/GROUPED[s]/"preds_cellsplicenet_released.csv")[["event_id","neuron","target","pred"]].merge(co,on="event_id",how="left")
    d["seen"]=d.event_id.isin(set(ev.index))
    d["lookup"]=d.event_id.map(ev).fillna(d.gene_id.map(gn)).fillna(gm)
    d["lookup_nr"]=d.lookup+d.neuron.map(noff).fillna(0)
    d["gbm"]=gbm.predict(feats(d))
    u=d[~d.seen]; allun.append(u.assign(split=s))
    print(f"\n== Drive ckpt on {s} test: {len(d)} rows / {d.event_id.nunique()} events; "
          f"seen in its training: {d[d.seen].event_id.nunique()} events, unseen: {u.event_id.nunique()} events ({len(u)} rows)")
    print(f"   {'':28s} pooled  within-event")
    for lab,c in [("Drive checkpoint","pred"),("event-mean lookup","lookup"),("lookup + neuron offset","lookup_nr")]:
        print(f"   {lab:28s} {sp(d,c):6.3f}  {wr(d,c):6.3f}")
u=pd.concat(allun).drop_duplicates(["event_id","neuron"])
print(f"\n== UNSEEN events pooled across all four test sets: {u.event_id.nunique()} events, {len(u)} rows")
for lab,c in [("Drive checkpoint","pred"),("gene-mean lookup","lookup"),("exon/intron-length GBM","gbm")]:
    print(f"   {lab:28s} pooled {sp(u,c):6.3f}")
print("   events per unseen:", u.groupby("event_id").size().describe()[["min","50%","max"]].to_dict())
