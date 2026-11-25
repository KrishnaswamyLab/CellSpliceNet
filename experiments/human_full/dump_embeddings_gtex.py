"""Dump CellSpliceNet latent (transformer_zrep) + metadata on the GTEx test split,
tissue-stratified subsample, for PHATE recoloring (reviewer response).

Mirrors test_eval.py setup. Env:
  CKPT             checkpoint .pth (default: latest gtex best_validation_model.pth)
  OUT_NPZ          output path
  PER_TISSUE       rows sampled per tissue (default 120 -> ~5880 points)
  BATCH_SIZE(8), SEED(42)
"""
import sys, os, pathlib, time, glob
import numpy as np

from experiments.common.paths import RUNS, split_config

BATCH_SIZE = os.environ.get("BATCH_SIZE", "8")
PER_TISSUE = int(os.environ.get("PER_TISSUE", "120"))
SEED       = int(os.environ.get("SEED", "42"))
OUT_NPZ    = os.environ["OUT_NPZ"]

default_ckpt = sorted(
    glob.glob(str(RUNS / "deltacellsplicenet" / "*" / "model" / "best_validation_model.pth")),
    key=os.path.getmtime,
)
CKPT = os.environ.get("CKPT", default_ckpt[-1] if default_ckpt else "")
if not CKPT or not os.path.exists(CKPT):
    raise SystemExit(f"No checkpoint. Set CKPT. (looked: {default_ckpt})")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.argv = [
    "dump_embeddings_gtex.py",
    "--config_fname",         str(split_config(os.environ.get("SPLIT", "gene"))),
    "--sfgenes",              "493",
    "--batch_size",           BATCH_SIZE,
    "--device",               "cuda",
]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from args import argparser_fn
from eval import repoint_data_paths
from data.splicedata_dataloader import splicedata_dataloader, sp3_collate_fn
from utils.validation import model_perd

args = argparser_fn(dataset_type="gtex", batch_size=int(BATCH_SIZE))
args.device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)
print(f"[dump] CKPT={CKPT} device={device} per_tissue={PER_TISSUE}", flush=True)

data = splicedata_dataloader(args); data.setup()
args = data.setup_hparams(args)

# tissue-stratified subsample of the test split. __getitem__ indexes
# filtered_df positionally (.iloc), so positional indices == dataset idx.
df = data.test_data.filtered_df.reset_index(drop=True)
rng = np.random.RandomState(SEED)
idx = []
for _tis, grp in df.groupby("neuron"):
    gi = grp.index.to_numpy()                       # positional after reset_index
    idx.extend(rng.choice(gi, size=min(PER_TISSUE, len(gi)), replace=False).tolist())
rng.shuffle(idx)
print(f"[dump] stratified {len(idx)} of {len(df)} rows across {df['neuron'].nunique()} tissues", flush=True)

loader = DataLoader(
    data.test_data, batch_size=int(BATCH_SIZE), sampler=SubsetRandomSampler(idx),
    num_workers=8, persistent_workers=True, pin_memory=True,
    collate_fn=sp3_collate_fn, drop_last=False,
)

model = torch.load(CKPT, weights_only=False, map_location=device)
if isinstance(model, nn.DataParallel):
    model = model.module
repoint_data_paths(model, args.dataset_paths, device)   # data files come from the config
model.to(device)
model.eval()

zreps, gids, eids, neurons, tgts, preds = [], [], [], [], [], []
t0 = time.time(); nb = len(loader)
with torch.no_grad():
    for it, (meta, x, y) in enumerate(loader):
        perds, _loss, embed_list = model_perd(model, x, meta, y, device, embed_list_return=True)
        z = embed_list["transformer_zrep"].detach().cpu().numpy()
        p = model.untransform_targets(perds).detach().cpu().squeeze(1).numpy()
        t = y["psi"].detach().cpu().squeeze(1).numpy()
        zreps.append(z)
        for m in meta:
            gids.append(m["gene_id"]); eids.append(m["event_id"]); neurons.append(m["neuron"])
        tgts.append(t); preds.append(p)
        if it % 50 == 0:
            print(f"[dump] batch {it}/{nb} n={len(eids)} ({time.time()-t0:.0f}s)", flush=True)

zrep = np.concatenate(zreps, 0)
np.savez_compressed(
    OUT_NPZ,
    zrep=zrep.astype(np.float32),
    gene_id=np.array(gids, dtype=object),
    event_id=np.array(eids, dtype=object),
    neuron=np.array(neurons, dtype=object),
    target=np.concatenate(tgts).astype(np.float32),
    pred=np.concatenate(preds).astype(np.float32),
)
print(f"[dump] wrote {OUT_NPZ}  zrep={zrep.shape}  n={len(eids)}  ({time.time()-t0:.0f}s)", flush=True)
