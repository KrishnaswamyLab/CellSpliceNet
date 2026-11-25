"""Pooled and within-event rho per checkpoint for the long-protocol grouped-split runs (full vs SeqOnly),
plus the per-neuron-offset reference (training-set neuron mean deviation) on the same test rows.
Predictions: dump_preds.py with CKPT_NAMES=all on <runs>/realexpr_long/* and <runs>/seqonly_long/* runs
(<preds>/<group>__<split>__seed<N>/preds_<ckpt>.csv)."""
import glob, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from experiments.common.paths import out_dir, read_split
from experiments.worm_ablations.common import GROUPED, read_preds

P = str(out_dir("worm_ablations", "preds")) + "/"
SPE = {"eventbased": 400, "genebased": 434, "neuronbased": 393}
def within(d, col):
    d = d[d.groupby("event_id").neuron.transform("count") >= 5]
    a = d.target - d.groupby("event_id").target.transform("mean"); b = d[col] - d.groupby("event_id")[col].transform("mean")
    return spearmanr(a, b).correlation if b.std() > 1e-9 else 0.0
off = {}
for spl in SPE:
    tr = read_split(GROUPED[spl], "train", "c_elegans")
    pc = [c for c in tr.columns if c.upper() == "PSI"][0]
    tr["dev"] = tr[pc] - tr.groupby("event_id")[pc].transform("mean"); off[spl] = tr.groupby("neuron").dev.mean()
rows = []
for root, lab in (("realexpr_long", "full"), ("seqonly_long", "seqonly")):
    for f in glob.glob(P + f"{root}__*__seed*/preds_*.csv"):
        run = os.path.basename(os.path.dirname(f)).split("__", 1)[1]; spl, seed = run.split("__seed")
        spl = {v: k for k, v in GROUPED.items()}.get(spl, spl)
        ck = os.path.basename(f)[6:-10]; m = re.match(r"step_(\d+)", ck)
        d = read_preds(f); d["off"] = d.neuron.map(off[spl]).fillna(0)
        rows.append(dict(model=lab, split=spl, seed=int(seed), ckpt=ck, epoch=round(int(m.group(1)) / SPE[spl]) if m else -1,
                         pooled=spearmanr(d.target, d.pred).correlation, within=within(d, "pred"), offset_within=within(d, "off")))
r = pd.DataFrame(rows); r.to_csv(out_dir("rebuttal") / "long_grouped_curve.csv", index=False)
pd.set_option("display.width", 200)
print(r.groupby(["split", "model", "epoch"])[["pooled", "within", "offset_within"]].agg(["mean", "std", "count"]).round(3).to_string())
