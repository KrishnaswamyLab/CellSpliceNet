"""Calibration for the nearest-tissue baseline: copy PSI from a RANDOM training tissue, and from the
nearest training tissue under centred expression similarity."""
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

from experiments.human_gen.common import SCATTER, baselines_file, read_split

X = str(SCATTER) + "/"
tr=read_split("cell","train",usecols=["event_id","neuron","PSI"]); te=pd.read_csv(baselines_file("cell")); te=te[te.gene_seen==1]
P=tr.pivot_table(index="event_id",columns="neuron",values="PSI")
tis=sorted(set(tr.neuron)|set(te.neuron)); A=np.stack([torch.load(X+f"scatter_coeffs_{t}.pt",map_location="cpu").flatten().float().numpy() for t in tis])
C=A-A.mean(0); C/=np.linalg.norm(C,axis=1,keepdims=True); Sim=pd.DataFrame(C@C.T,index=tis,columns=tis)
def within(d,c):
    d=d[d.groupby("event_id").neuron.transform("count")>=5]
    a=d.PSI-d.groupby("event_id").PSI.transform("mean"); b=d[c]-d.groupby("event_id")[c].transform("mean")
    return spearmanr(a,b).correlation
def copy_from(src):  # src: dict test tissue -> ordered list of training tissues
    out=[]
    for e,t in zip(te.event_id,te.neuron):
        row=P.loc[e] if e in P.index else None; v=np.nan
        if row is not None:
            for s in src[t]:
                if not np.isnan(row[s]): v=row[s]; break
        out.append(v)
    return np.array(out)
trt=list(P.columns); rng=np.random.default_rng(0)
te["nn_centred"]=copy_from({t:list(Sim.loc[t,trt].sort_values(ascending=False).index) for t in te.neuron.unique()})
for k in range(3):
    te[f"random{k}"]=copy_from({t:list(rng.permutation(trt)) for t in te.neuron.unique()})
for c in ["nn_tissue","knn3_tissue","nn_centred","random0","random1","random2"]:
    d=te.dropna(subset=[c]); print(f"{c:12s} within={within(d,c):.3f} pooled={spearmanr(d.PSI,d[c]).correlation:.3f}",flush=True)
print("nearest (centred):",{t:Sim.loc[t,trt].idxmax() for t in te.neuron.unique()})
