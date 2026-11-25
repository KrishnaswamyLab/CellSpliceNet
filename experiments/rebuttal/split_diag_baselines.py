"""Diagnostics of the row-level and grouped C. elegans splits: overlap with training, variance decomposition, and
lookup / length-feature baselines next to the full model (dump_preds.py predictions of <runs>/fullmodel_SE32k/SE
and <runs>/fullmodel_grouped)."""
import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

from experiments.common.paths import split_csv
from experiments.worm_ablations.common import EVENTS_COORDS, GROUPED, read_preds, run_preds

SPL={"row-level": ("SE", "fullmodel_SE32k")}
for s in ["eventbased","genebased","neuronbased"]:
    SPL[s]=(s, "fullmodel_grouped")

def feats(df):
    f=pd.DataFrame(index=df.index)
    f["exon_len"]=df.exon_end-df.exon_start
    f["up_intron"]=df.exon_start-df.intron_start
    f["dn_intron"]=df.intron_end-df.exon_end
    f["gene_len"]=df.gene_length
    f["rel_pos"]=df.exon_start/df.gene_length
    f["exon_mod3"]=(f.exon_len%3==0).astype(int)
    return f

def resid_rho(df,col):
    g=df.groupby("event_id")
    k=g.neuron.transform("count")>=5
    d=df[k]
    rt=d.target-d.groupby("event_id").target.transform("mean")
    rp=d[col]-d.groupby("event_id")[col].transform("mean")
    return spearmanr(rt,rp).correlation if rp.std()>0 else 0.0

def sp(a,b): return spearmanr(a,b).correlation

for name,(spl,group) in SPL.items():
    tr=pd.read_csv(split_csv(GROUPED[spl],"train","c_elegans")); te=pd.read_csv(split_csv(GROUPED[spl],"test","c_elegans"))
    tr=tr.rename(columns={"PSI":"psi"}); te=te.rename(columns={"PSI":"psi"})
    CC=["exon_start","exon_end","intron_start","intron_end","gene_length"]
    if "exon_start" not in tr.columns:
        co=pd.read_csv(EVENTS_COORDS,sep="\t")[["event_id"]+CC].drop_duplicates("event_id")
        tr=tr.merge(co,on="event_id",how="left"); te=te.merge(co,on="event_id",how="left")
        if "gene_id" not in tr.columns:
            g=pd.read_csv(EVENTS_COORDS,sep="\t")[["event_id","gene_id"]].drop_duplicates("event_id")
            tr=tr.merge(g,on="event_id",how="left"); te=te.merge(g,on="event_id",how="left")
    print(f"\n===== {name} =====")
    print(f"train: {tr.event_id.nunique()} events, {tr.gene_id.nunique()} genes, {tr.neuron.nunique()} neurons, {len(tr)} rows")
    print(f"test : {te.event_id.nunique()} events, {te.gene_id.nunique()} genes, {te.neuron.nunique()} neurons, {len(te)} rows")
    se=set(tr.event_id); sg=set(tr.gene_id); sn=set(tr.neuron)
    print(f"test events seen in train: {te.event_id.isin(se).mean():.0%} of rows | genes seen: {te.gene_id.isin(sg).mean():.0%} | neurons seen: {te.neuron.isin(sn).mean():.0%}")
    # variance decomposition on test
    em=te.groupby("event_id").psi.transform("mean")
    tot=((te.psi-te.psi.mean())**2).sum(); within=((te.psi-em)**2).sum()
    print(f"test variance: between-event {1-within/tot:.1%}, within-event {within/tot:.1%}")
    # baselines
    gmean=tr.psi.mean()
    ev=tr.groupby("event_id").psi.mean(); gn=tr.groupby("gene_id").psi.mean(); nr=tr.groupby("neuron").psi.mean()
    tr["eres"]=tr.psi-tr.event_id.map(ev); noff=tr.groupby("neuron").eres.mean()
    te["b_event"]=te.event_id.map(ev).fillna(te.gene_id.map(gn)).fillna(gmean)
    te["b_gene"]=te.gene_id.map(gn).fillna(gmean)
    te["b_neuron"]=te.neuron.map(nr).fillna(gmean)
    te["b_event_neuron"]=te.b_event+te.neuron.map(noff).fillna(0)
    # feature GBM trained on event-level means
    tre=tr.groupby("event_id").agg(psi=("psi","mean"),exon_start=("exon_start","first"),exon_end=("exon_end","first"),
        intron_start=("intron_start","first"),intron_end=("intron_end","first"),gene_length=("gene_length","first"))
    gbm=GradientBoostingRegressor(random_state=0,max_depth=3,n_estimators=200,learning_rate=0.05).fit(feats(tre),tre.psi)
    te["b_lengths"]=gbm.predict(feats(te))
    # merge model preds (3 seeds)
    mp=[]
    for s in (1,2,3):
        try: p=read_preds(run_preds(group,spl,s))[["event_id","neuron","target","pred"]]
        except FileNotFoundError: continue
        p=p.rename(columns={"pred":f"m{s}"}).drop(columns="target")
        mp.append(p)
    df=te.copy()
    for p in mp: df=df.merge(p,on=["event_id","neuron"],how="inner")
    df["target"]=df.psi
    rows=[("event-mean lookup (train)","b_event"),("gene-mean lookup","b_gene"),("neuron-mean only","b_neuron"),
          ("event mean + neuron offset","b_event_neuron"),("exon/intron-length GBM","b_lengths")]
    rows+=[(f"our full model seed{c[1]}",c) for c in df.columns if c in ("m1","m2","m3")]
    print(f"scored on {len(df)} rows / {df.event_id.nunique()} events (rows the model was evaluated on)")
    print(f"  {'predictor':32s} pooled_rho  within-event_rho")
    for lab,c in rows:
        print(f"  {lab:32s} {sp(df.target,df[c]):9.3f}  {resid_rho(df,c):9.3f}")
