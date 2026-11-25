"""Evaluate ONE GTEx checkpoint on a fixed, representative VALID subsample.

Why this exists: GTEx training (full_train.py / train_full.py) selected
best_validation_model.pth on a capped (VALID_MAX_BATCHES=200 -> ~1600 rows)
*shuffled* validation pass, redrawn every eval. With a 282,695-row valid set
that is ~0.57% of valid, redrawn each time -> noisy model selection. This script
recomputes the native-space valid loss (the same model_perd loss training used
to pick "best") on a FIXED seeded subsample so checkpoints can be re-ranked and
the genuine best re-selected. See memory: csn-partial-validation-artifact.

Like test_eval.py, the valid CSV is tissue-sorted, so we use a seeded random
sampler (not the first N rows). VALID_MAX_SAMPLES=0 -> full valid.

Env vars:
  CKPT               path to .pth (required)
  RESULTS_CSV        shared CSV to append one row to (default <out>/human_full/gtex_valid_reselect.csv)
  SPLIT              human split (default gene)
  RUN_LABEL          run name for the row (default: parent-of-model dirname)
  BATCH_SIZE         default 8
  VALID_MAX_SAMPLES  default 20000 (~7% of valid; seeded, tissue-balanced). 0 = full.
  SEED               default 42 (fixes the subsample so all ckpts see the SAME rows)
"""
import sys, os, pathlib, time, csv
import numpy as np

from experiments.common.paths import out_dir, split_config

CKPT             = os.environ["CKPT"]
RESULTS_CSV      = pathlib.Path(os.environ.get("RESULTS_CSV") or out_dir("human_full") / "gtex_valid_reselect.csv")
BATCH_SIZE       = os.environ.get("BATCH_SIZE", "8")
VALID_MAX_SAMPLES = int(os.environ.get("VALID_MAX_SAMPLES", "20000"))
SEED             = int(os.environ.get("SEED", "42"))
RUN_LABEL        = os.environ.get("RUN_LABEL", pathlib.Path(CKPT).parents[1].name)
CKPT_NAME        = pathlib.Path(CKPT).name

if not os.path.exists(CKPT):
    raise SystemExit(f"[gtex_valid_eval] no checkpoint: {CKPT}")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.argv = [
    "gtex_valid_eval.py",
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

data = splicedata_dataloader(args); data.setup()
args = data.setup_hparams(args)
n_valid = len(data.valid_data)

if VALID_MAX_SAMPLES > 0 and VALID_MAX_SAMPLES < n_valid:
    rng = np.random.RandomState(SEED)
    indices = rng.choice(n_valid, size=VALID_MAX_SAMPLES, replace=False).tolist()
    n_eval = VALID_MAX_SAMPLES
    print(f"[gtex_valid_eval] sampling {VALID_MAX_SAMPLES} of {n_valid} valid rows (seed={SEED})", flush=True)
else:
    indices = list(range(n_valid))
    n_eval = n_valid
    print(f"[gtex_valid_eval] evaluating all {n_valid} valid rows", flush=True)

valid_loader = DataLoader(
    data.valid_data,
    batch_size=int(BATCH_SIZE),
    sampler=SubsetRandomSampler(indices),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=sp3_collate_fn,
    drop_last=False,
)

model = torch.load(CKPT, weights_only=False, map_location=device)
if isinstance(model, nn.DataParallel):
    model = model.module
repoint_data_paths(model, args.dataset_paths, device)   # data files come from the config
model.to(device)
# checkpoints baked self.device='cuda' on submodules; override to eval device
for _m in model.modules():
    if hasattr(_m, "device"):
        _m.device = device
model.eval()

print(f"[gtex_valid_eval] run={RUN_LABEL} ckpt={CKPT_NAME} n_eval={n_eval}", flush=True)

t0 = time.time()
losses, preds_all, targets_all = [], [], []
nb = len(valid_loader)
with torch.no_grad():
    for it, (meta, x, y) in enumerate(valid_loader):
        preds, loss_i, _ = model_perd(model, x, meta, y, device, embed_list_return=False)
        losses.append(float(loss_i.detach().cpu().item()))
        preds_all.append(model.untransform_targets(preds).detach().cpu().squeeze(1))
        targets_all.append(y["psi"].detach().cpu().squeeze(1))
        if it % 200 == 0:
            print(f"[gtex_valid_eval]   batch {it}/{nb} elapsed={time.time()-t0:.0f}s", flush=True)

preds_t  = torch.cat(preds_all).numpy()
target_t = torch.cat(targets_all).numpy()
valid_loss = float(np.mean(losses)) if losses else float("nan")
mse = float(((preds_t - target_t) ** 2).mean())
mae = float(np.abs(preds_t - target_t).mean())
try:
    from scipy.stats import pearsonr, spearmanr
    pearson  = float(pearsonr(preds_t, target_t).statistic)
    spearman = float(spearmanr(preds_t, target_t).statistic)
except Exception:
    pearson = float(np.corrcoef(preds_t, target_t)[0, 1]); spearman = float("nan")

print(f"[gtex_valid_eval] {RUN_LABEL}/{CKPT_NAME}: n={len(preds_t)} "
      f"valid_loss={valid_loss:.4f} mse={mse:.4f} mae={mae:.4f} "
      f"pearson={pearson:.4f} spearman={spearman:.4f} ({time.time()-t0:.0f}s)", flush=True)

write_header = not RESULTS_CSV.exists()
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_CSV, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(["run", "ckpt", "n_eval", "valid_max_samples", "seed",
                    "valid_loss", "valid_mse", "valid_mae", "valid_pearson", "valid_spearman"])
    w.writerow([RUN_LABEL, CKPT_NAME, len(preds_t), VALID_MAX_SAMPLES, SEED,
                valid_loss, mse, mae, pearson, spearman])
print(f"[gtex_valid_eval] appended row -> {RESULTS_CSV}", flush=True)
