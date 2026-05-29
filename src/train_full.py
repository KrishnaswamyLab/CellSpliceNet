"""Step-based trainer (the recommended entry point).

Why step-based: large datasets (e.g. GTEx, ~2.18M rows) make an "epoch"
impractically large, so training is budgeted in optimizer steps and wallclock
time rather than epochs. The legacy epoch-based loop lives in ``train.py`` and
is kept only for the small worm dataset / reproducibility.

This script is path-agnostic: every data path comes from CLI flags (see
``args.py``), so a colleague can clone the repo and point it at data anywhere:

    python src/train_full.py \
        --config_fname        /path/to/data_config.ini \
        --expression_data_root /path/to/train_data.csv \
        --structure_data_root /path/to/structure_scattering_dict.pkl \
        --scatter_coeffs_dir  /path/to/scatter_coeffs \
        --sfgenes 493 --batch_size 4 --n_steps 200000

With no flags it falls back to the defaults in ``args.py`` (data under
``<repo>/dataset/``). Outputs go to ``<repo>/outputs/<model>/<key><type>/``.

Run-budget knobs are CLI flags below; each also accepts an env-var fallback so
existing sbatch wrappers that export N_STEPS / EVAL_EVERY / etc. keep working.
"""
import os
import sys
import time
import json
import argparse

# Make `from args import ...` work regardless of the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from args import argparser_fn, print_gpu_info, make_directroy
from data.splicedata_dataloader import splicedata_dataloader
from utils.validation import model_perd
from viz.training_scatter import scatter_plot
from model import model_registry


def _env(name, default):
    v = os.environ.get(name)
    return v if v is not None else default


def parse_run_control():
    """Run-budget / logging knobs not already in args.py. CLI flags win;
    env vars (used by the sbatch wrappers) are the fallback defaults. Parsed
    with parse_known_args so the model/data flags pass through to argparser_fn.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data_tag", default=_env("DATASET_TAG", "gtex"),
                   help="Short dataset tag; expands to dataset_type '01Feb2025_<tag>'.")
    p.add_argument("--time_budget_s", type=float,
                   default=float(_env("TIME_BUDGET_S", str(5 * 3600 + 50 * 60))),
                   help="Wallclock budget in seconds; stop cleanly before SLURM walltime.")
    p.add_argument("--valid_max_batches", type=int,
                   default=int(_env("VALID_MAX_BATCHES", "200")),
                   help="Cap validation iterations so each eval stays short.")
    p.add_argument("--save_step_ckpt_every", type=int,
                   default=int(_env("SAVE_STEP_CKPT_EVERY", "10")),
                   help="Save a step checkpoint every Nth validation (best+final always saved). 0 disables.")
    p.add_argument("--log_every_iters", type=int,
                   default=int(_env("LOG_EVERY_ITERS", "50")))
    p.add_argument("--log_every_secs", type=float,
                   default=float(_env("LOG_EVERY_SECS", "30")))
    rc, _ = p.parse_known_args()
    return rc


def quick_validate(model, loader, max_batches, device):
    """Capped-iteration validation. Returns (mean_loss, preds, targets)."""
    model.eval()
    losses, preds_list, targets_list = [], [], []
    with torch.no_grad():
        for it, (meta, x, y) in enumerate(loader):
            if it >= max_batches:
                break
            preds, loss_i, _ = model_perd(model, x, meta, y, device, embed_list_return=False)
            losses.append(loss_i.detach().cpu().item())
            unwrapped = model.module if isinstance(model, nn.DataParallel) else model
            psi_preds = unwrapped.untransform_targets(preds)
            preds_list.append(psi_preds.detach().cpu().squeeze(1))
            targets_list.append(y["psi"].detach().cpu().squeeze(1))
    return (float(np.mean(losses)) if losses else float("nan"),
            torch.cat(preds_list) if preds_list else torch.empty(0),
            torch.cat(targets_list) if targets_list else torch.empty(0))


def main():
    rc = parse_run_control()

    # argparser_fn reads the remaining flags from sys.argv (paths, sfgenes,
    # lr, n_steps, eval_every_iteration, device, batch_size, ...).
    args = argparser_fn(dataset_type=rc.data_tag)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    data = splicedata_dataloader(args)
    data.setup()
    train_loader = data.train_dataloader()
    valid_loader = data.valid_dataloader()
    args = data.setup_hparams(args)
    print(f"[full] device={args.device} sfgenes={args.sfgenes} "
          f"train_N={getattr(data, 'train_N', '?')} valid_N={getattr(data, 'valid_N', '?')} "
          f"test_N={getattr(data, 'test_N', '?')}", flush=True)

    ModelClass = model_registry.str2model(args.model)
    model = ModelClass(args)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[full] using {torch.cuda.device_count()} GPUs via DataParallel", flush=True)
        model = nn.DataParallel(model)
    model.to(device)

    # Portable output dir: <repo>/outputs/<model>/<key><dataset_type>/ (+ subdirs).
    args.output_key = time.strftime(args.model + "--%Y%m%d-%H%M%S-")
    args.output_dir = make_directroy(args)
    print_gpu_info(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    n_steps = args.n_steps
    eval_every = args.eval_every_iteration
    loss_dict = {"train_loss": [], "valid_loss": [], "step": []}
    best_valoss = np.inf
    prt_eval = ("[full] step %6d | train loss (avg-since-last): %4.3f | "
                "valid loss: %4.3f | GPU: %.0f MB | elapsed: %.0fs")

    print(f"[full] step-based loop: n_steps={n_steps}, time_budget={rc.time_budget_s:.0f}s "
          f"({rc.time_budget_s / 3600:.1f}h), eval_every={eval_every}, "
          f"valid_max_batches={rc.valid_max_batches}", flush=True)

    t_start = time.time()
    t_last_log = t_start
    step = 0
    trlosses_window = []
    done = False

    while step < n_steps and not done:
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

            if (step % rc.log_every_iters == 0) or (now - t_last_log >= rc.log_every_secs):
                elapsed = now - t_start
                rate = step / max(elapsed, 1e-6)
                remaining = max(rc.time_budget_s - elapsed, 0)
                print(f"[full] step {step}/{n_steps} loss={loss_i.item():.4f} "
                      f"gpu={gpu_mb:.0f}MB rate={rate:.2f} it/s "
                      f"elapsed={elapsed:.0f}s budget_left={remaining:.0f}s", flush=True)
                t_last_log = now

            if step % eval_every == 0:
                train_loss = float(np.mean(trlosses_window)) if trlosses_window else float("nan")
                trlosses_window = []
                valid_loss, preds_v, targets_v = quick_validate(
                    model, valid_loader, rc.valid_max_batches, device)
                if preds_v.numel() > 0:
                    scatter_plot(
                        gt_tensor=targets_v, pred_tensor=preds_v,
                        path=args.output_dir + f"/scatter_valid/psi/psi_at_step_{step}",
                        title="", label="", dpi=100, scatter_color="#0077b6", title_add=True,
                    )
                print(prt_eval % (step, train_loss, valid_loss, gpu_mb, time.time() - t_start), flush=True)

                if valid_loss < best_valoss:
                    best_valoss = valid_loss
                    torch.save(model.to("cpu"), args.output_dir + "/model/best_validation_model.pth")
                    model.to(device)

                loss_dict["train_loss"].append(train_loss)
                loss_dict["valid_loss"].append(valid_loss)
                loss_dict["step"].append(step)
                with open(args.output_dir + "/model/loss_dict.json", "w") as f:
                    json.dump(loss_dict, f)

                n_valid = len(loss_dict["step"])
                if rc.save_step_ckpt_every > 0 and n_valid % rc.save_step_ckpt_every == 0:
                    torch.save(model.to("cpu"), args.output_dir + f"/model/step_{step}_model.pth")
                    model.to(device)
                scheduler.step()

            if step >= n_steps or (time.time() - t_start) >= rc.time_budget_s:
                done = True
                break

    elapsed = time.time() - t_start
    print(f"[full] training stopped: step={step} elapsed={elapsed:.0f}s "
          f"({elapsed / 3600:.2f}h) best_valid_loss={best_valoss:.4f}", flush=True)
    torch.save(model.to("cpu"), args.output_dir + "/model/final_model.pth")
    print(f"[full] saved final checkpoint to {args.output_dir}/model/", flush=True)


if __name__ == "__main__":
    main()
