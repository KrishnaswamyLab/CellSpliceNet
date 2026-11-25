"""Expression-swap test on the released Drive checkpoint (row-level SE test set).

The checkpoint (CellSpliceNet README -> Drive 'CellSpliceNet.pth') is a 243-splice-factor
deltacellsplicenet. It pickles with our CellSpliceNet/src, but its stored data paths point
at /gpfs/radev and its expression module carries `scatter_path` where our code reads `scatter_dir`, so both
are repointed at the data root (src/eval.py repoint_data_paths).

Neuron identity reaches the model only through the expression input, so the test reassigns neuron
expression profiles (random derangements, most / least similar neuron, one constant neuron) and reports
pooled and within-event Spearman on the rows_se test set.

Env: CKPT (default: the released checkpoint in the data root), OUT_DIR (default <out>/rebuttal/released), BATCH_SIZE
"""
import os, sys, json, time, pathlib
import numpy as np

from experiments.common.paths import out_dir, released_checkpoint, split_config

CKPT = os.environ.get("CKPT") or str(released_checkpoint())
OUT_DIR = pathlib.Path(os.environ.get("OUT_DIR") or out_dir("rebuttal", "released"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, torch.nn as nn
from scipy.stats import rankdata
from args import argparser_fn
from data.splicedata_dataloader import splicedata_dataloader
from eval import repoint_data_paths

def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    return float(((rx-rx.mean())*(ry-ry.mean())).mean()/(rx.std()*ry.std())) if rx.std() and ry.std() else float("nan")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Neuron identity reaches the model ONLY through the expression module
# (metadata["neuron"] -> scatter_coeffs_<neuron>.pt), so all within-event variation in the
# predictions comes from expression. This test asks whether the CORRECT neuron's profile
# matters: reassign profiles at random, to the most/least similar neuron, or to one neuron.
import pandas as pd, csv
args_list = ["expr_swap_released.py",
    "--config_fname", str(split_config("rows_se", "c_elegans")), "--structure_length", "500",
    "--expression_encoder", "scatter", "--sfgenes", "243",
    "--batch_size", str(BATCH_SIZE), "--lr", "2e-5", "--device", str(device), "--eval_every_iteration", "500"]
sys.argv = args_list
args = argparser_fn(dataset_type="worm_drive_ckpt", batch_size=BATCH_SIZE)
args.device = str(device)
data = splicedata_dataloader(args); data.setup(); args = data.setup_hparams(args)
model = torch.load(CKPT, weights_only=False, map_location=device)
if isinstance(model, nn.DataParallel): model = model.module
repoint_data_paths(model, args.dataset_paths, device)
SCATTER = str(args.dataset_paths.scatter_coeffs_dir)
model.to(device)
for _m in model.modules():
    if hasattr(_m, "device"): _m.device = device
model.eval()

batches = list(data.test_dataloader())
neurons = sorted({m["neuron"] for b in batches for m in b[0]})
print(f"[swap] test batches={len(batches)} neurons={len(neurons)}", flush=True)

# neuron-neuron similarity in the model's own expression input space
V = []
for n in neurons:
    t = torch.load(pathlib.Path(SCATTER) / f"scatter_coeffs_{n}.pt", map_location="cpu", weights_only=False)
    V.append(torch.as_tensor(t).detach().to(torch.float64).flatten().numpy())
V = np.stack(V); V = V - V.mean(0, keepdims=True)
V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
S = V @ V.T; np.fill_diagonal(S, np.nan)
most = {n: neurons[int(np.nanargmax(S[i]))] for i, n in enumerate(neurons)}
least = {n: neurons[int(np.nanargmin(S[i]))] for i, n in enumerate(neurons)}
print(f"[swap] mean similarity: to most-similar {np.nanmean(np.nanmax(S,1)):.3f}, "
      f"to least-similar {np.nanmean(np.nanmin(S,1)):.3f}, random pair {np.nanmean(S):.3f}", flush=True)

def derangement(seed):
    r = np.random.default_rng(seed)
    while True:
        p = r.permutation(len(neurons))
        if not np.any(p == np.arange(len(neurons))):
            return {n: neurons[p[i]] for i, n in enumerate(neurons)}

conds = [("clean", None)]
conds += [(f"random_perm_{s}", derangement(s)) for s in range(1, 6)]
conds += [("most_similar", most), ("least_similar", least)]
conds += [(f"constant_{n}", {k: n for k in neurons}) for n in neurons[:3]]

def within(df):
    d = df[df.groupby("event_id").neuron.transform("count") >= 5]
    a = d.target - d.groupby("event_id").target.transform("mean")
    b = d.pred - d.groupby("event_id").pred.transform("mean")
    return spearman(b.values, a.values) if b.std() > 1e-9 else 0.0

out_rows = []
for name, mp in conds:
    rows = []; t0 = time.time()
    with torch.no_grad():
        for meta, x, y in batches:
            m2 = meta if mp is None else [dict(m, neuron=mp[m["neuron"]]) for m in meta]
            p = model([t.to(device) for t in x], m2, embed_list_return=False)
            if isinstance(p, tuple): p = p[0]
            pr = model.untransform_targets(p).detach().cpu().squeeze(1).numpy()
            tg = y["psi"].detach().cpu().squeeze(1).numpy()
            rows += [(m["event_id"], m["neuron"], float(tg[i]), float(pr[i])) for i, m in enumerate(meta)]
    df = pd.DataFrame(rows, columns=["event_id", "neuron", "target", "pred"])
    df.to_csv(OUT_DIR / f"exprswap_SE_test__{name}.csv", index=False)
    r = dict(condition=name, n=len(df), pooled=spearman(df.pred.values, df.target.values), within_event=within(df))
    out_rows.append(r)
    print(f"[swap] {name:22s} pooled={r['pooled']:.4f} within_event={r['within_event']:.4f} ({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame(out_rows).to_csv(OUT_DIR / "exprswap_SE_test_summary.csv", index=False)
print("[swap] done", flush=True)
