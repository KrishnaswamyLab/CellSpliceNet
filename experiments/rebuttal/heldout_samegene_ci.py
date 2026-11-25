import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
"""Event-held-out split: does the full model beat a same-gene mean lookup on test exons whose gene has other
training exons? (dump_preds.py predictions of <runs>/fullmodel_grouped/event_sample1__seed{1,2,3})"""
from scipy.stats import spearmanr
from experiments.common.paths import out_dir, read_split
from experiments.worm_ablations.common import EVENTS_COORDS, GENE_IDS, GROUPED, SC_EXPR, read_preds, run_preds
tr=read_split("event_sample1","train","c_elegans"); gmean=tr.groupby("gene_id").PSI.mean()
te=read_split("event_sample1","test","c_elegans")[["event_id","gene_id"]].drop_duplicates("event_id")
P=[read_preds(run_preds("fullmodel_grouped","eventbased",s))[["event_id","neuron","target","pred"]].merge(te,on="event_id") for s in (1,2,3)]
sp=lambda a,b: spearmanr(a,b).correlation
for s,p in enumerate(P,1):
    x=p[p.gene_id.isin(gmean.index)]; y=p[~p.gene_id.isin(gmean.index)]
    print(f"seed{s}: seen-gene rho {sp(x.target,x.pred):.3f}   unseen-gene rho {sp(y.target,y.pred):.3f}")
x=P[0][["event_id","neuron","target","gene_id"]].copy(); x["model"]=np.mean([p.pred.values for p in P],0)
x=x[x.gene_id.isin(gmean.index)]; x["gm"]=x.gene_id.map(gmean)
ev=x.event_id.unique(); g={e:v for e,v in x.groupby("event_id")}; r=np.random.default_rng(0); out=[]
for _ in range(2000):
    s=pd.concat([g[e] for e in r.choice(ev,len(ev))]); out.append((sp(s.target,s.model),sp(s.target,s.gm)))
o=np.array(out)
print(f"seen-gene ({len(ev)} events): model {sp(x.target,x.model):.3f} CI {np.percentile(o[:,0],[2.5,97.5]).round(3)} | gene-mean lookup {sp(x.target,x.gm):.3f} CI {np.percentile(o[:,1],[2.5,97.5]).round(3)} | diff CI {np.percentile(o[:,0]-o[:,1],[2.5,97.5]).round(3)}")
print("genes with >1 event in whole SE set:", (pd.concat([tr,read_split("event_sample1","test","c_elegans")]).groupby('gene_id').event_id.nunique()>1).mean().round(3))
