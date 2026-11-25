import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
"""Grouped-split test accuracy stratified by knowledge-graph similarity of the test gene to training genes.
Needs the C. elegans KG2E gene embeddings (KG_DIR, default <data_root>/c_elegans/raw/kg: entities_kg_emb_2023-03-10_
celegans_KG2E_nepochs_20.npy, entity2labeldict_...npy) and dump_preds.py predictions of <runs>/fullmodel_grouped."""
import os
from scipy.stats import spearmanr
from experiments.common.paths import out_dir, read_split
from experiments.worm_ablations.common import EVENTS_COORDS, GENE_IDS, GROUPED, SC_EXPR, read_preds, run_preds
from experiments.common.paths import WORM
KG = os.environ.get("KG_DIR", str(WORM / "raw/kg")) + "/"
E=np.load(KG+"entities_kg_emb_2023-03-10_celegans_KG2E_nepochs_20.npy")
L=np.load(KG+"entity2labeldict_2023-03-10_celegans_KG2E_nepochs_20.npy",allow_pickle=True).item()
En=E/np.linalg.norm(E,axis=1,keepdims=True)
def emb(g): i=L.get(g); return None if i is None else En[i]
def sp(a,b): return spearmanr(a,b).correlation if len(a)>5 else np.nan
def boot(df,col,n=1000,seed=0):
    ev=df.event_id.unique(); r=np.random.default_rng(seed); g={e:x for e,x in df.groupby("event_id")}; out=[]
    for _ in range(n):
        s=pd.concat([g[e] for e in r.choice(ev,len(ev))]); out.append(sp(s.target,s[col]))
    return np.nanpercentile(out,[2.5,97.5])
for split in ["eventbased","genebased"]:
    tr=read_split(GROUPED[split],"train","c_elegans")
    evm=tr.groupby(["event_id","gene_id"]).PSI.mean().reset_index()
    trg=sorted(set(tr.gene_id)); TG=np.array([emb(g) for g in trg if emb(g) is not None]); tgn=[g for g in trg if emb(g) is not None]
    gmean=tr.groupby("gene_id").PSI.mean()
    preds=[read_preds(run_preds("fullmodel_grouped",split,s))[["event_id","neuron","target","pred"]].rename(columns={"pred":f"m{s}"}) for s in (1,2,3)]
    df=preds[0]
    for p in preds[1:]: df=df.merge(p.drop(columns="target"),on=["event_id","neuron"])
    df["model"]=df[["m1","m2","m3"]].mean(1)
    te=read_split(GROUPED[split],"test","c_elegans")[["event_id","gene_id"]].drop_duplicates("event_id")
    df=df.merge(te,on="event_id",how="left")
    # KG nearest-training-gene similarity and kNN prediction (k=5 genes, similarity-weighted gene means)
    sims={}; knn={}
    for g in df.gene_id.unique():
        v=emb(g)
        if v is None: continue
        s=TG@v; idx=np.argsort(-s)[:5]; w=np.clip(s[idx],0,None)+1e-6
        sims[g]=s.max(); knn[g]=np.average([gmean[tgn[i]] for i in idx],weights=w)
    df["kg_sim"]=df.gene_id.map(sims); df["kg_knn"]=df.gene_id.map(knn)
    df["gene_in_train"]=df.gene_id.isin(set(tr.gene_id))
    print(f"\n===== {split} test: {df.event_id.nunique()} events, {len(df)} rows; model = mean of 3 seeds (pooled rho {sp(df.target,df.model):.3f}) =====")
    if df.gene_in_train.any():
        for flag,lab in [(True,"gene has other events in train"),(False,"gene unseen")]:
            x=df[df.gene_in_train==flag]
            print(f"  {lab:32s} {x.event_id.nunique():4d} ev  model rho {sp(x.target,x.model):6.3f}  gene-mean lookup {sp(x.target,x.gene_id.map(gmean)) if flag else float('nan'):6.3f}")
    x=df.dropna(subset=["kg_sim"])
    print(f"  KG-embedded subset: {x.event_id.nunique()} ev; KG-kNN predictor rho {sp(x.target,x.kg_knn):.3f} CI {np.round(boot(x,'kg_knn'),3)} | model on same rows {sp(x.target,x.model):.3f}")
    ev_sim=x.groupby("event_id").kg_sim.first()
    q=pd.qcut(ev_sim,3,labels=["low","mid","high"]); x=x.assign(tier=x.event_id.map(q))
    for t in ["low","mid","high"]:
        y=x[x.tier==t]
        print(f"   KG similarity tier {t:4s} (max cos {y.kg_sim.min():.2f}-{y.kg_sim.max():.2f}): {y.event_id.nunique():3d} ev  model {sp(y.target,y.model):6.3f}  KG-kNN {sp(y.target,y.kg_knn):6.3f}")
