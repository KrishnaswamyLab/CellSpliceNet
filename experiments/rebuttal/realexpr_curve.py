"""Pooled and within-event rho vs training length for the real-expression runs.

Reads preds_<ckpt>.csv (dump_preds.py, CKPT_NAMES=all) for every checkpoint of the long
(Arman-protocol, batch 16, 400 epochs) runs and the short (batch 8, 32k-step) runs.
Within-event rho = Spearman of pooled per-event residuals (events with >= 5 neurons).
Reference points on the same test sets:
  row-level   : event-mean lookup pooled 0.880; lookup + neuron offset within-event 0.222;
                released checkpoints within-event 0.21-0.40 (mean 0.30)
  neuron-held-out: event-mean lookup pooled 0.825; a lookup has NO information about a new
                neuron, so its within-event rho is 0 -> anything > 0 is new-neuron generalisation
"""
import glob, os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from experiments.common.paths import out_dir
from experiments.worm_ablations.common import GROUPED, read_preds

P = str(out_dir("worm_ablations", "preds")) + "/"
STEPS_PER_EPOCH = {"SE": 462, "neuronbased": 393}


def within(d):
    d = d[d.groupby("event_id").neuron.transform("count") >= 5]
    a = d.target - d.groupby("event_id").target.transform("mean")
    b = d.pred - d.groupby("event_id").pred.transform("mean")
    return spearmanr(a, b).correlation if b.std() > 1e-9 else 0.0


rows = []
for kind in ("realexpr_long", "realexpr"):
    for f in glob.glob(P + f"{kind}__*__seed*/preds_*.csv"):
        run = os.path.basename(os.path.dirname(f)).split("__", 1)[1]; spl, seed = run.split("__seed")
        spl = {v: k for k, v in GROUPED.items()}.get(spl, spl)
        ck = os.path.basename(f)[len("preds_"):-len("_model.csv")]
        m = re.match(r"step_(\d+)", ck)
        epoch = round(int(m.group(1)) / STEPS_PER_EPOCH[spl]) if m else None
        d = read_preds(f)
        rows.append(dict(kind=kind, split=spl, seed=int(seed), ckpt=ck, epoch=epoch,
                         pooled=spearmanr(d.target, d.pred).correlation, within=within(d)))
r = pd.DataFrame(rows)
r.to_csv(out_dir("rebuttal") / "realexpr_curve_all_ckpts.csv", index=False)

pd.set_option("display.width", 200)
for spl in ("SE", "neuronbased"):
    x = r[r.split == spl]
    print(f"\n===== {spl} =====")
    g = x[x.kind == "realexpr_long"]
    curve = (g[g.epoch.notna()].groupby("epoch")[["pooled", "within"]].agg(["mean", "std", "count"]).round(3))
    print("long runs (batch 16), by epoch:\n", curve.to_string())
    for kind, lab in (("realexpr_long", "long, best-valid ckpt"), ("realexpr", "short 32k, best-valid ckpt")):
        b = x[(x.kind == kind) & (x.ckpt == "best_validation")]
        if len(b):
            print(f"{lab:28s} n={len(b)}  pooled {b.pooled.mean():.3f} +/- {b.pooled.std():.3f}"
                  f"   within {b.within.mean():.3f} +/- {b.within.std():.3f}   (seeds {sorted(b.seed)})")
