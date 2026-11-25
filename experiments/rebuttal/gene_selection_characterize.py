"""How do the SE-event genes compare with other neuron-expressed genes (expression breadth, level, HVG)?
Writes <out>/rebuttal/gene_char.csv (used by heldout_gene_subset_eval.py)."""
import pandas as pd, numpy as np, scipy.io, scipy.sparse as sp
from experiments.common.paths import out_dir, read_split
from experiments.worm_ablations.common import EVENTS_COORDS, GENE_IDS, GROUPED, SC_EXPR, read_preds, run_preds
E=str(SC_EXPR)+"/"
X=scipy.io.mmread(E+"241119b_scCounts_neuronsBulk.mtx").tocsc()
genes=[l.strip() for l in open(E+"241119b_scCounts_neuronsBulk.rownames")]
cells=[l.strip() for l in open(E+"241119b_scCounts_neuronsBulk.colnames")]
ctype=np.array([c.rsplit("_",1)[0] for c in cells]); types=sorted(set(ctype))
# pseudobulk CPM per neuron type
M=np.zeros((len(genes),len(types)))
for j,t in enumerate(types):
    M[:,j]=np.asarray(X[:,ctype==t].sum(1)).ravel()
cpm=M/M.sum(0,keepdims=True)*1e6
# detection fraction per type (fraction of cells with >0 counts)
Xb=(X>0).astype(np.float32).tocsc()
D=np.zeros_like(M)
for j,t in enumerate(types):
    idx=np.where(ctype==t)[0]; D[:,j]=np.asarray(Xb[:,idx].mean(1)).ravel()
g=pd.DataFrame(index=genes)
g["mean_logcpm"]=np.log1p(cpm).mean(1)
g["breadth"]=(D>=0.1).mean(1)          # fraction of neuron types where >=10% of cells detect it
g["expressed"]=(D>=0.1).any(1)
# Seurat-style HVG: dispersion = var/mean of CPM across types, z-scored within 20 mean bins
mu=cpm.mean(1); var=cpm.var(1); disp=np.log(var/(mu+1e-12)+1e-12)
g["logmean"]=np.log1p(mu); g["disp"]=disp
ex=g[g.expressed].copy()
ex["bin"]=pd.qcut(ex.logmean,20,labels=False,duplicates="drop")
ex["disp_z"]=ex.groupby("bin").disp.transform(lambda s:(s-s.mean())/s.std())
ex["hvg2000"]=ex.disp_z.rank(ascending=False)<=2000
names=pd.read_csv(GENE_IDS,header=None,names=["tax","gid","name","seq","status","biotype"]).set_index("gid")
ex["name"]=names.name.reindex(ex.index).fillna(names.seq.reindex(ex.index))
ex["biotype"]=names.biotype.reindex(ex.index)
co=pd.read_csv(EVENTS_COORDS,sep="\t")
se=pd.concat([read_split("rows_se",s,"c_elegans") for s in ["train","valid","test"]])
ours=set(se.gene_id)
print(f"genome: {(names.biotype=='protein_coding_gene').sum()} live protein-coding gene IDs in WS291; "
      f"{len(genes)} genes in CeNGEN neuron matrix; {len(ex)} detected in >=1 neuron type (of {len(types)} types)")
print(f"our SE genes: {len(ours)}; found in matrix: {len(ours & set(ex.index))}")
ex["ours"]=ex.index.isin(ours)
pc=ex[ex.biotype=="protein_coding_gene"]
def summ(d,lab):
    print(f"{lab:38s} n={len(d):6d}  breadth median={d.breadth.median():.2f}  "
          f"broad(>=80% types)={ (d.breadth>=0.8).mean():.0%}  restricted(<=20%)={(d.breadth<=0.2).mean():.0%}  "
          f"mean log-CPM median={d.mean_logcpm.median():.2f}  HVG-top2000={d.hvg2000.mean():.0%}")
summ(pc[~pc.ours],"other neuron-expressed protein-coding")
summ(pc[pc.ours],"our 529 SE genes")
ex.to_csv(out_dir("rebuttal")/"gene_char.csv")
o=pc[pc.ours].sort_values("disp_z",ascending=False)
print("\nmost variable of ours:", ", ".join(o.name.head(30)))
print("\nleast variable (broadest) of ours:", ", ".join(o.sort_values('breadth',ascending=False).name.head(30)))
