"""Evaluate one trained worm-ablation checkpoint on the valid + test splits.

Loads RUN_DIR/model/best_validation_model.pth, rebuilds the dataloader with the
exact inputs that run was trained on (read from RUN_DIR/ablation_config.json),
runs full valid + full test passes, and computes loss / Pearson / Spearman on
untransformed PSI. Appends one row to RESULTS_CSV (created with header if absent).

Why this exists: the ablation training driver (worm_train.py) logs only a
regression loss; Spearman was computed inside scatter_plot but discarded (only
burned into PNG titles). Test was never run at all. This recovers both from the
saved best checkpoints.

Env vars:
  RUN_DIR       ablation output dir (must contain model/best_validation_model.pth
                and ablation_config.json)
  RESULTS_CSV   shared CSV to append a row to (default: <out>/worm_ablations/eval_metrics.csv)
  SPLIT         runs made before the repo: data-root split they were trained on (default rows)
  CKPT_NAME     which checkpoint (default: best_validation_model.pth)
  BATCH_SIZE    default from config (else 8)
  DEVICE        cuda | cpu (default cuda if available)
"""
import sys, os, pathlib, json, csv, time
import numpy as np

from experiments.common.paths import out_dir
from experiments.worm_ablations.common import run_argv, run_config

RUN_DIR = pathlib.Path(os.environ["RUN_DIR"]).resolve()
CKPT_NAME = os.environ.get("CKPT_NAME", "best_validation_model.pth")
CKPT = RUN_DIR / "model" / CKPT_NAME
CFG_FILE = RUN_DIR / "ablation_config.json"
RESULTS_CSV = pathlib.Path(os.environ.get("RESULTS_CSV", str(out_dir("worm_ablations") / "eval_metrics.csv")))

if not CKPT.exists():
    raise SystemExit(f"[worm_eval] no checkpoint: {CKPT}")
if not CFG_FILE.exists():  # runs made by worm_train.py
    raise SystemExit(f"[worm_eval] no ablation_config.json: {CFG_FILE}")

cfg, DATA_CONFIG = run_config(RUN_DIR)   # older runs: data config derived from $SPLIT (default rows)
config_name = RUN_DIR.name  # e.g. magic_mi_knn3_scatter__seed1337
axis_dir = RUN_DIR.parent.name  # expression | structure
BATCH_SIZE = os.environ.get("BATCH_SIZE", str(cfg.get("batch_size", 8)))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.argv = run_argv("worm_eval.py", cfg, DATA_CONFIG, BATCH_SIZE)

import torch
import torch.nn as nn

# NOTE: seed AFTER imports (utils.validation clobbers the seed at import; see
# worm_train.py). Eval is deterministic anyway (shuffle=False), but keep it tidy.
from args import argparser_fn
from data.splicedata_dataloader import splicedata_dataloader
from eval import repoint_data_paths
from utils.validation import model_perd

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(DEVICE)

args = argparser_fn(dataset_type=cfg.get("dataset_tag", "worm_ablation"), batch_size=int(BATCH_SIZE))
args.device = DEVICE

data = splicedata_dataloader(args)
data.setup()
valid_loader = data.valid_dataloader()
test_loader = data.test_dataloader()
args = data.setup_hparams(args)

print(f"[worm_eval] run={config_name} encoder={cfg['expression_encoder']} "
      f"metric={cfg.get('expression_metric')} stc_len={cfg['structure_len']} "
      f"device={device} valid_N={data.valid_N} test_N={data.test_N}", flush=True)

model = torch.load(str(CKPT), weights_only=False, map_location=device)
if isinstance(model, nn.DataParallel):
    model = model.module
repoint_data_paths(model, args.dataset_paths, device)   # data files come from the config, not the pickle
model.to(device)
# The checkpoint baked in self.device='cuda' (set at train time) on the top
# model and submodules (e.g. ztransformer); override so .to(self.device) calls
# follow the eval device.
for _m in model.modules():
    if hasattr(_m, "device"):
        _m.device = device
model.eval()


# DUMP_PREDS=1 -> also write a per-row CSV (event_id,neuron,true_psi,pred_psi)
# next to RESULTS_CSV, keyed on metadata (NOT positional CSV alignment -- the
# test loader filters ~1000 rows). meta[i] is the meta_data DICT from
# data/dataloader.py:__getitem__ (keys 'event_id','neuron',...).
DUMP_PREDS = os.environ.get("DUMP_PREDS", "0") == "1"


def evaluate(loader, split):
    losses, preds_list, targets_list = [], [], []
    rows_out = []
    t0 = time.time()
    with torch.no_grad():
        for it, (meta, x, y) in enumerate(loader):
            preds, loss_i, _ = model_perd(model, x, meta, y, device, embed_list_return=False)
            losses.append(float(loss_i.detach().cpu().item()))
            psi_preds = model.untransform_targets(preds).detach().cpu().squeeze(1)
            preds_list.append(psi_preds)
            targets_list.append(y["psi"].detach().cpu().squeeze(1))
            if DUMP_PREDS:
                pp = psi_preds.tolist()
                tt = y["psi"].detach().cpu().squeeze(1).tolist()
                for i, m in enumerate(meta):
                    rows_out.append((m["event_id"], m["neuron"], tt[i], pp[i]))
    preds_t = torch.cat(preds_list).numpy()
    target_t = torch.cat(targets_list).numpy()
    if DUMP_PREDS:
        dump = RESULTS_CSV.parent / f"{config_name}__{split}_preds.csv"
        with open(dump, "w", newline="") as fh:
            wd = csv.writer(fh)
            wd.writerow(["event_id", "neuron", "true_psi", "pred_psi"])
            wd.writerows(rows_out)
        print(f"[worm_eval]   dumped {len(rows_out)} {split} rows -> {dump}", flush=True)
    loss = float(np.mean(losses)) if losses else float("nan")
    mse = float(((preds_t - target_t) ** 2).mean())
    mae = float(np.abs(preds_t - target_t).mean())
    try:
        from scipy.stats import pearsonr, spearmanr
        pearson = float(pearsonr(preds_t, target_t).statistic)
        spearman = float(spearmanr(preds_t, target_t).statistic)
    except Exception:
        pearson = float(np.corrcoef(preds_t, target_t)[0, 1])
        spearman = float("nan")
    print(f"[worm_eval]   {split}: n={len(preds_t)} loss={loss:.4f} mse={mse:.4f} "
          f"mae={mae:.4f} pearson={pearson:.4f} spearman={spearman:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return dict(n=len(preds_t), loss=loss, mse=mse, mae=mae, pearson=pearson, spearman=spearman)


v = evaluate(valid_loader, "valid")
t = evaluate(test_loader, "test")

row = {
    "config": config_name,
    "axis": axis_dir,
    "encoder": cfg["expression_encoder"],
    "metric": cfg.get("expression_metric"),
    "scatter_subdir": cfg.get("scatter_subdir"),
    "structure_len": cfg["structure_len"],
    "seed_label": cfg.get("seed"),
    "valid_loss": v["loss"], "valid_pearson": v["pearson"], "valid_spearman": v["spearman"],
    "valid_mse": v["mse"], "valid_mae": v["mae"], "valid_n": v["n"],
    "test_loss": t["loss"], "test_pearson": t["pearson"], "test_spearman": t["spearman"],
    "test_mse": t["mse"], "test_mae": t["mae"], "test_n": t["n"],
}
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
write_header = not RESULTS_CSV.exists()
with open(RESULTS_CSV, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)
print(f"[worm_eval] appended row for {config_name} -> {RESULTS_CSV}", flush=True)
