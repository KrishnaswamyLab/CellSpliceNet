"""Evaluate GTEx checkpoints on the SAME seeded random subsample of a human split's test rows (dataloader built
once, models reloaded in a loop -> comparable): seed replicates (confidence intervals), checkpoints along
training (calibration sweep), or the data-size runs (SPLIT=gene_ev500 etc. evaluate on the gene test set).

Writes per-row preds to OUTDIR/preds_<tag>.csv, one metrics row per ckpt to OUTDIR/ci_summary.csv, and the
aggregate mean+/-SD to OUTDIR/ci_stats.txt.
Env: CKPTS (':'-separated .pth paths), OUTDIR (default <out>/human_full/subsample_eval), SPLIT (default gene),
     TEST_MAX_SAMPLES(=20000), SEED(=42), BATCH_SIZE(=8). The tag of a checkpoint is <run dir>/<ckpt name>.
"""
import os, sys, re, time, pathlib, csv
import numpy as np

from experiments.common.paths import out_dir, split_config

BATCH_SIZE = os.environ.get("BATCH_SIZE", "8")
TEST_MAX_SAMPLES = int(os.environ.get("TEST_MAX_SAMPLES", "20000"))
SEED = int(os.environ.get("SEED", "42"))
CKPTS = [c for c in os.environ["CKPTS"].split(":") if c]
OUTDIR = os.environ.get("OUTDIR") or str(out_dir("human_full", "subsample_eval"))
os.makedirs(OUTDIR, exist_ok=True)
CI_SUMMARY = os.path.join(OUTDIR, "ci_summary.csv")
for c in CKPTS:
    assert os.path.exists(c), c


def ckpt_tag(path):
    p = pathlib.Path(path)
    return f"{p.parents[1].name}__{p.stem}"


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.argv = [
    "eval_gtex_ci.py",
    "--config_fname",         str(split_config(os.environ.get("SPLIT", "gene"))),
    "--sfgenes",              "493",
    "--batch_size",           BATCH_SIZE,
    "--device",               "cuda",
]
import torch, torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from args import argparser_fn
from eval import repoint_data_paths
from data.splicedata_dataloader import splicedata_dataloader, sp3_collate_fn
from utils.validation import model_perd
from scipy.stats import pearsonr, spearmanr

args = argparser_fn(dataset_type="gtex", batch_size=int(BATCH_SIZE))
args.device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)
data = splicedata_dataloader(args); data.setup(); args = data.setup_hparams(args)
n_test = len(data.test_data)
rng = np.random.RandomState(SEED)
indices = rng.choice(n_test, size=min(TEST_MAX_SAMPLES, n_test), replace=False).tolist()
loader = DataLoader(data.test_data, batch_size=int(BATCH_SIZE),
                    sampler=SubsetRandomSampler(indices), num_workers=8,
                    persistent_workers=True, pin_memory=True,
                    collate_fn=sp3_collate_fn, drop_last=False)

spearmans, pearsons = [], []
for CKPT in CKPTS:
    tag = ckpt_tag(CKPT)
    model = torch.load(CKPT, weights_only=False, map_location=device)
    if isinstance(model, nn.DataParallel): model = model.module
    repoint_data_paths(model, args.dataset_paths, device)   # data files come from the config
    model.to(device); model.eval()
    preds_all, targ_all, meta_all = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for meta, x, y in loader:
            preds, _, _ = model_perd(model, x, meta, y, device, embed_list_return=False)
            preds_all.append(model.untransform_targets(preds).detach().cpu().squeeze(1))
            targ_all.append(y["psi"].detach().cpu().squeeze(1))
            for m in meta: meta_all.append((m["gene_id"], m["event_id"], m["neuron"]))
    p = torch.cat(preds_all).numpy(); t = torch.cat(targ_all).numpy()
    pe = float(pearsonr(p, t).statistic); sp = float(spearmanr(p, t).statistic)
    r2 = float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))
    bias = float((p - t).mean()); mse = float(((p - t) ** 2).mean())
    spearmans.append(sp); pearsons.append(pe)
    print(f"[ci] tag={tag} n={len(p)} Pearson={pe:.4f} Spearman={sp:.4f} "
          f"R2cod={r2:.4f} bias={bias:+.4f} mse={mse:.4f} ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(OUTDIR, f"preds_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gene_id", "event_id", "neuron", "psi_target", "psi_pred"])
        for m, tt, pp in zip(meta_all, t.tolist(), p.tolist()):
            w.writerow([m[0], m[1], m[2], tt, pp])
    row = dict(tag=tag, n=len(p), pearson=round(pe, 4), spearman=round(sp, 4),
               r2cod=round(r2, 4), mean_bias=round(bias, 4), mse=round(mse, 4), ckpt=CKPT)
    hdr = not os.path.exists(CI_SUMMARY)
    with open(CI_SUMMARY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if hdr: w.writeheader()
        w.writerow(row)

n = len(spearmans)
sp_arr = np.array(spearmans); pe_arr = np.array(pearsons)
# sample SD (ddof=1) for reporting over independent seed replicates
sp_sd = float(sp_arr.std(ddof=1)) if n > 1 else 0.0
pe_sd = float(pe_arr.std(ddof=1)) if n > 1 else 0.0
stats = (f"GTEx CSN test Spearman: {sp_arr.mean():.4f} +/- {sp_sd:.4f} (n={n})\n"
         f"GTEx CSN test Pearson:  {pe_arr.mean():.4f} +/- {pe_sd:.4f} (n={n})\n"
         f"per-seed Spearman: {[round(x,4) for x in spearmans]}\n"
         f"per-seed Pearson:  {[round(x,4) for x in pearsons]}\n")
print("[ci] " + stats.replace("\n", "\n[ci] "), flush=True)
with open(os.path.join(OUTDIR, "ci_stats.txt"), "w") as f:
    f.write(stats)
print(f"[ci] done -> {CI_SUMMARY}")
