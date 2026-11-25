"""Worm-data training driver for the expression/structure ablation matrix.

One process = one ablation variant. The variant is selected entirely via
environment variables (see below) so the same script can be submitted for every
variant by launch.sh (through train.sbatch) without code edits.

Env vars (all optional; defaults reproduce the scatter/magic/mi_knn3/500bp anchor):
  SPLIT                    data-root split (default rows; old names SE / eventbased / ... accepted)
                           or a data_config.ini; CONFIG_FNAME is accepted as an alias
  MODEL                    model registry name (default deltacellsplicenet)
  EXPRESSION_ENCODER       scatter | mlp | mlp_node | mlp_scalar | gnn
  EXPRESSION_ABLATION_ROOT denoiser dir (default <data_root>/c_elegans/processed/expression/ablations/magic)
  EXPRESSION_METRIC        subdir under ablation root (mi_knn3, pearson, ...)
                           ignored for encoder=mlp
  STRUCTURE_LEN            structure window length (300 | 500 | 750 | 1000); STRUCTURE_PKL overrides the file
  STRUCTURE_SCALING        fixed (default) | batch (the scaling used by runs trained before Nov 2026)
  OUTPUT_DIR               where to write checkpoints + loss_dict.json + data_config.ini
                           (default <runs>/default, see experiments/worm_ablations/common.py)
  DATASET_TAG              suffix for run id (default: worm_ablation)
  BATCH_SIZE               default 8
  N_STEPS                  default 60000
  TIME_BUDGET_S            wallclock budget (seconds); default ~5h50m
  EVAL_EVERY               default 500
  VALID_MAX_BATCHES        cap on validation iters (default 200)
  SAVE_STEP_CKPT_EVERY     0 = never save step ckpts (default 0 for ablations)
  SEED                     default 42

Train loop is step-based + capped on wallclock, same shape as full_train.py.
"""
import sys, os, pathlib, time, json
import numpy as np

from experiments.common.paths import derive_config
from experiments.worm_ablations.common import EXPR_ABL, abl_runs, legacy_path, split_ini, structure_pkl

EXP_ENCODER = os.environ.get("EXPRESSION_ENCODER", "scatter").lower()
EXP_ROOT    = legacy_path(os.environ.get("EXPRESSION_ABLATION_ROOT", str(EXPR_ABL / "magic")))
EXP_METRIC  = os.environ.get("EXPRESSION_METRIC", "mi_knn3")

STC_LEN = int(os.environ.get("STRUCTURE_LEN", "500"))
STC_PKL = legacy_path(os.environ.get("STRUCTURE_PKL", str(structure_pkl(STC_LEN))))

DATASET_TAG = os.environ.get("DATASET_TAG", "worm_ablation")
OUT_DIR = pathlib.Path(os.environ.get("OUTPUT_DIR", str(abl_runs() / "default")))
SPLIT = os.environ.get("SPLIT") or os.environ.get("CONFIG_FNAME") or "rows"

BATCH_SIZE        = os.environ.get("BATCH_SIZE", "8")
N_STEPS           = int(os.environ.get("N_STEPS", "60000"))
TIME_BUDGET_S     = float(os.environ.get("TIME_BUDGET_S", str(5 * 3600 + 50 * 60)))
EVAL_EVERY        = int(os.environ.get("EVAL_EVERY", "500"))
VALID_MAX_BATCHES = int(os.environ.get("VALID_MAX_BATCHES", "200"))
SAVE_STEP_CKPT_EVERY = int(os.environ.get("SAVE_STEP_CKPT_EVERY", "0"))
LOG_EVERY_ITERS   = int(os.environ.get("LOG_EVERY_ITERS", "50"))
LOG_EVERY_SECS    = float(os.environ.get("LOG_EVERY_SECS", "30"))
SEED              = int(os.environ.get("SEED", "42"))

# Which scatter-coeff sibling dir to use. Defaults to the legacy unweighted
# full-graph dir. New options live next to it (binary density sweep +
# continuous_weighted) and are drop-in: same per-celltype <CELL>.pt files,
# same (242,11,242) float32 shape. Only meaningful difference for the
# pearson-vs-spearman axis is in these new dirs (legacy scatter_coeffs is
# byte-identical pearson==spearman -- the known artifact).
SCATTER_SUBDIR = os.environ.get("SCATTER_SUBDIR", "scatter_coeffs")
scatter_dir = f"{EXP_ROOT}/{EXP_METRIC}/{SCATTER_SUBDIR}"
# SCATTER_DIR overrides the above, e.g. the real 243-factor per-neuron expression
# (<data_root>/c_elegans/processed/expression/scatter_243); set SFGENES=243 with it.
scatter_dir = legacy_path(os.environ.get("SCATTER_DIR", scatter_dir))
SFGENES     = os.environ.get("SFGENES", "242")
mean_dir    = f"{EXP_ROOT}/mean_vecs"
graph_dir   = f"{EXP_ROOT}/{EXP_METRIC}/graphs"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# The run's data config: the split's files plus this variant's structure / expression inputs.
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FNAME = str(derive_config(split_ini(SPLIT), OUT_DIR / "data_config.ini", structure_data_root=STC_PKL,
                                 scatter_coeffs_dir=scatter_dir, mean_vec_dir=mean_dir, graph_metric_dir=graph_dir))

sys.argv = [
    "worm_train.py",
    "--config_fname",         CONFIG_FNAME,
    "--structure_length",     str(STC_LEN),
    "--expression_encoder",   EXP_ENCODER,
    "--sfgenes",              SFGENES,
    "--batch_size",           str(BATCH_SIZE),
    "--lr",                   "2e-5",
    "--device",               "cuda",
    "--eval_every_iteration", str(EVAL_EVERY),
]

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from args import argparser_fn, print_gpu_info
from data.splicedata_dataloader import splicedata_dataloader
from utils.validation import model_perd
from viz.training_scatter import scatter_plot
from model import model_registry

# NOTE: must seed AFTER the imports above. utils/validation.py runs a
# module-level torch.manual_seed(1337)/np.random.seed(42) at import time, which
# clobbers any seed set before the import -- that bug made every SEED value
# produce bit-identical runs (no real error bars). Seeding here, right before
# the shuffle=True dataloader and model init, is what actually varies the run.
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

args = argparser_fn(dataset_type=DATASET_TAG, batch_size=int(BATCH_SIZE))   # --model defaults to $MODEL
args.device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)

data = splicedata_dataloader(args)
data.setup()
train_loader = data.train_dataloader()
valid_loader = data.valid_dataloader()
args = data.setup_hparams(args)
print(f"[ablation] encoder={EXP_ENCODER} root={EXP_ROOT} metric={EXP_METRIC} "
      f"stc_len={STC_LEN} stc_pkl={STC_PKL}", flush=True)
print(f"[ablation] sfgenes={args.sfgenes} train_N={data.train_N} "
      f"valid_N={data.valid_N} test_N={data.test_N} seed={SEED}", flush=True)

ModelClass = model_registry.str2model(args.model)
model = ModelClass(args).to(device)

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "model").mkdir(exist_ok=True)
(OUT_DIR / "scatter_valid/psi").mkdir(parents=True, exist_ok=True)
args.output_dir = str(OUT_DIR)
args.output_key = time.strftime(args.model + "--%Y%m%d-%H%M%S-")
print_gpu_info(args)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


def quick_validate(model, loader, max_batches, device):
    model.eval()
    losses, preds_list, targets_list = [], [], []
    with torch.no_grad():
        for it, (meta, x, y) in enumerate(loader):
            if it >= max_batches:
                break
            preds, loss_i, _ = model_perd(model, x, meta, y, device, embed_list_return=False)
            losses.append(loss_i.detach().cpu().item())
            psi_preds = model.module.untransform_targets(preds) if isinstance(model, nn.DataParallel) \
                        else model.untransform_targets(preds)
            preds_list.append(psi_preds.detach().cpu().squeeze(1))
            targets_list.append(y["psi"].detach().cpu().squeeze(1))
    return (float(np.mean(losses)) if losses else float("nan"),
            torch.cat(preds_list) if preds_list else torch.empty(0),
            torch.cat(targets_list) if targets_list else torch.empty(0))


# Persist the ablation config for reproducibility
with open(OUT_DIR / "ablation_config.json", "w") as f:
    json.dump({
        "expression_encoder": EXP_ENCODER,
        "expression_ablation_root": EXP_ROOT,
        "expression_metric": EXP_METRIC,
        "split": SPLIT,
        "config_fname": CONFIG_FNAME,
        "model": args.model,
        "structure_scaling": args.structure_scaling,
        "scatter_subdir": SCATTER_SUBDIR,
        "scatter_coeffs_dir": scatter_dir,
        "mean_vec_dir": mean_dir,
        "graph_metric_dir": graph_dir,
        "structure_pkl": STC_PKL,
        "structure_len": STC_LEN,
        "seed": SEED,
        "n_steps": N_STEPS,
        "time_budget_s": TIME_BUDGET_S,
        "batch_size": int(BATCH_SIZE),
        "lr": args.lr,
        "sfgenes": args.sfgenes,
        "dataset_tag": DATASET_TAG,
    }, f, indent=2)

loss_dict = {"train_loss": [], "valid_loss": [], "step": []}
best_valoss = np.inf
prt_eval = "[ablation] step %6d | train loss: %4.3f | valid loss: %4.3f | GPU: %.0f MB | elapsed: %.0fs"
print(f"[ablation] step-based loop: n_steps={N_STEPS}, time_budget={TIME_BUDGET_S:.0f}s "
      f"({TIME_BUDGET_S/3600:.1f}h), eval_every={EVAL_EVERY}", flush=True)

t_start = time.time()
t_last_log = t_start
step = 0
trlosses_window = []
done = False

while step < N_STEPS and not done:
    for (meta, x, y) in train_loader:
        model.train()
        preds, loss_i, _ = model_perd(model, x, meta, y, device, embed_list_return=False)
        optimizer.zero_grad()
        loss_i.backward()
        optimizer.step()
        trlosses_window.append(loss_i.item())
        step += 1

        now = time.time()
        gpu_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

        if (step % LOG_EVERY_ITERS == 0) or (now - t_last_log >= LOG_EVERY_SECS):
            elapsed = now - t_start
            rate = step / max(elapsed, 1e-6)
            remaining = max(TIME_BUDGET_S - elapsed, 0)
            print(f"[ablation] step {step}/{N_STEPS} loss={loss_i.item():.4f} "
                  f"gpu={gpu_mb:.0f}MB rate={rate:.2f} it/s "
                  f"elapsed={elapsed:.0f}s budget_left={remaining:.0f}s", flush=True)
            t_last_log = now

        if step % EVAL_EVERY == 0:
            train_loss = float(np.mean(trlosses_window)) if trlosses_window else float("nan")
            trlosses_window = []
            valid_loss, preds_v, targets_v = quick_validate(model, valid_loader, VALID_MAX_BATCHES, device)
            if preds_v.numel() > 0:
                scatter_plot(
                    gt_tensor=targets_v, pred_tensor=preds_v,
                    path=str(OUT_DIR / f"scatter_valid/psi/psi_at_step_{step}"),
                    title="", label="", dpi=100, scatter_color="#0077b6", title_add=True,
                )
            print(prt_eval % (step, train_loss, valid_loss, gpu_mb, time.time() - t_start), flush=True)

            if valid_loss < best_valoss:
                best_valoss = valid_loss
                torch.save(model.to("cpu"), str(OUT_DIR / "model/best_validation_model.pth"))
                model.to(device)

            loss_dict["train_loss"].append(train_loss)
            loss_dict["valid_loss"].append(valid_loss)
            loss_dict["step"].append(step)
            with open(OUT_DIR / "model/loss_dict.json", "w") as f:
                json.dump(loss_dict, f)

            n_valid = len(loss_dict["step"])
            if SAVE_STEP_CKPT_EVERY > 0 and n_valid % SAVE_STEP_CKPT_EVERY == 0:
                torch.save(model.to("cpu"), str(OUT_DIR / f"model/step_{step}_model.pth"))
                model.to(device)
            scheduler.step()

        if step >= N_STEPS or (time.time() - t_start) >= TIME_BUDGET_S:
            done = True
            break

elapsed = time.time() - t_start
print(f"[ablation] done: step={step} elapsed={elapsed:.0f}s "
      f"({elapsed/3600:.2f}h) best_valid_loss={best_valoss:.4f}", flush=True)
torch.save(model.to("cpu"), str(OUT_DIR / "model/final_model.pth"))
print(f"[ablation] wrote final checkpoint to {OUT_DIR}/model/", flush=True)
