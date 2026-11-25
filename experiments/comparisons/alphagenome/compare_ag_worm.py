"""Zero-shot AlphaGenome vs CellSpliceNet vs simple baselines on worm SE test sets (same rows)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from experiments.common.paths import out_dir, split_csv
from experiments.worm_ablations.common import GROUPED, read_preds, run_preds
ag = pd.read_csv(out_dir("comparisons", "alphagenome") / "ag_worm_psi.csv")
ag["all3"] = ag.found_up & ag.found_dn & ag.found_ex
sp = lambda a, b: spearmanr(a, b, nan_policy="omit").correlation
def feats(d):
    return pd.DataFrame({"el": d.exon_end - d.exon_start, "up": d.exon_start - d.intron_start, "dn": d.intron_end - d.exon_end,
                         "gl": d.gene_length, "rel": d.exon_start / d.gene_length, "m3": ((d.exon_end - d.exon_start) % 3 == 0).astype(int)})
CC = ["exon_start", "exon_end", "intron_start", "intron_end", "gene_length"]
co = pd.concat([pd.read_csv(split_csv("event_sample1", s, "c_elegans")) for s in ("train", "valid", "test")])[["event_id"] + CC].drop_duplicates("event_id")
SPL = {"row-level": (split_csv("rows_se", "train", "c_elegans"), ("fullmodel_SE32k", "SE"))}
for s in ("eventbased", "genebased", "neuronbased"):
    SPL[s] = (split_csv(GROUPED[s], "train", "c_elegans"), ("fullmodel_grouped", s))
for name, (trf, pf) in SPL.items():
    tr = pd.read_csv(trf).drop(columns=[c for c in CC if c in pd.read_csv(trf, nrows=1).columns], errors="ignore").merge(co, on="event_id")
    te = tre = None
    tre = tr.groupby("event_id").agg(psi=("PSI", "mean"), **{c: (c, "first") for c in CC})
    gbm = GradientBoostingRegressor(random_state=0, max_depth=3, n_estimators=200, learning_rate=0.05).fit(feats(tre), tre.psi)
    evm = tr.groupby("event_id").PSI.mean()
    P = [read_preds(run_preds(pf[0], pf[1], s))[["event_id", "neuron", "target", "pred"]] for s in (1, 2, 3)]
    d = P[0][["event_id", "neuron", "target"]].copy()
    for i, p in enumerate(P, 1):
        d = d.merge(p.rename(columns={"pred": f"m{i}"}).drop(columns="target"), on=["event_id", "neuron"])
    d = d.merge(co, on="event_id").merge(ag, on="event_id", how="left")
    d["gbm"] = gbm.predict(feats(d)); d["lookup"] = d.event_id.map(evm)
    n_ag = d.ag_psi_all.notna().mean()
    print(f"\n===== {name} test: {d.event_id.nunique()} events / {len(d)} rows; AlphaGenome scored {n_ag:.0%} of rows, all-3-junctions {d.all3.mean():.0%} =====")
    def line(x, lab):
        csn = [sp(x.target, x[f"m{i}"]) for i in (1, 2, 3)]
        print(f"  {lab:28s} ev={x.event_id.nunique():3d}  CSN {np.mean(csn):.3f}+/-{np.std(csn, ddof=1):.3f} | AG(all GTEx) {sp(x.target, x.ag_psi_all):.3f}"
              f"  AG(brain) {sp(x.target, x.ag_psi_brain):.3f} | length-GBM {sp(x.target, x.gbm):.3f}"
              + (f" | event-mean lookup {sp(x.target, x.lookup):.3f}" if x.lookup.notna().mean() > 0.5 else ""))
    line(d[d.ag_psi_all.notna()], "rows AG could score")
    line(d[d.all3 == True], "AG found all 3 junctions")
