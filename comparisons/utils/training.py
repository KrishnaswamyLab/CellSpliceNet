"""Sample-budgeted training loop for comparison baselines."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from tqdm import tqdm

from log_utils import log

PredictFn = Callable[[torch.nn.Module, object, torch.device], tuple[torch.Tensor, torch.Tensor]]

REF_BATCH_SIZE = 64
DEFAULT_EVAL_EVERY_SAMPLES = REF_BATCH_SIZE * 500


def default_n_samples(data_tag: str) -> int:
    """Default sample budget (ref_steps * REF_BATCH_SIZE). Override with N_SAMPLES env."""
    if "N_SAMPLES" in os.environ:
        return int(os.environ["N_SAMPLES"])
    if data_tag.lower() in ("gtex", "human"):
        ref_steps = int(os.environ.get("N_STEPS", "200000"))
    else:
        ref_steps = int(os.environ.get("N_STEPS", "10000"))
    return ref_steps * REF_BATCH_SIZE


def add_comparison_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=2e-5, type=float)
    parser.add_argument("--data-tag", default="replicate", type=str)
    parser.add_argument("--random-seed", default=1, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--n-samples", default=50000, type=int, help="Training sample budget (total examples seen; batch-size independent).")
    parser.add_argument("--eval-every", default=int(os.environ.get("EVAL_EVERY_SAMPLES", str(DEFAULT_EVAL_EVERY_SAMPLES))), type=int, help="Run capped validation every N training samples.")
    parser.add_argument("--valid-max-batches", default=int(os.environ.get("VALID_MAX_BATCHES", "200")), type=int, help="Max validation batches per eval (train_full.py default).")
    parser.add_argument("--time-budget-s", default=None, type=float, help="Optional wallclock budget in seconds.")


def resolve_n_samples(data_tag: str, n_samples: Optional[int]) -> int:
    return n_samples if n_samples is not None else default_n_samples(data_tag)


def _batch_size(data_item) -> int:
    return int(data_item[2]["psi"].shape[0])


def metrics_from_arrays(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> tuple[float, float, float]:
    pearson_R = pearsonr(y_true_arr, y_pred_arr)[0]
    spearman_R = spearmanr(a=y_true_arr, b=y_pred_arr)[0]
    r2 = r2_score(y_true_arr, y_pred_arr)
    return pearson_R, spearman_R, r2


def evaluate_loader(
    model: torch.nn.Module,
    loader,
    predict_fn: PredictFn,
    loss_fn: torch.nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> tuple[float, float, float, float]:
    model.eval()
    losses: list[float] = []
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, data_item in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            y_pred, y_true = predict_fn(model, data_item, device)
            y_pred = torch.sigmoid(y_pred)
            loss = loss_fn(y_true, y_pred)
            losses.append(loss.mean().item())
            y_true_parts.append(y_true.flatten().detach().cpu().numpy())
            y_pred_parts.append(y_pred.flatten().detach().cpu().numpy())

    if not losses:
        return float("nan"), float("nan"), float("nan"), float("nan")

    y_true_arr = np.hstack(y_true_parts)
    y_pred_arr = np.hstack(y_pred_parts)
    mean_loss = float(np.mean(losses))
    pearson_R, spearman_R, r2 = metrics_from_arrays(y_true_arr, y_pred_arr)
    return mean_loss, pearson_R, spearman_R, r2


def run_step_training(
    *,
    model: torch.nn.Module,
    data,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: torch.nn.Module,
    predict_fn: PredictFn,
    log_file,
    model_save_path,
    n_samples: int,
    eval_every: int,
    valid_max_batches: int,
    time_budget_s: Optional[float] = None,
    method_name: str = "comparison",
) -> None:
    train_loader = data.train_dataloader(shuffle_bool=True)
    valid_loader = data.valid_dataloader(shuffle_bool=False)
    test_loader = data.test_dataloader(shuffle_bool=False)

    train_N = getattr(data, "train_N", "?")
    log(
        f"[{method_name}] sample-budget training: n_samples={n_samples}, eval_every={eval_every} samples, "
        f"valid_max_batches={valid_max_batches}, train_N={train_N}",
        filepath=str(log_file),
    )

    best_val_loss = np.inf
    step = 0
    samples_seen = 0
    samples_at_last_eval = 0
    train_loss_window: list[float] = []
    train_iter = iter(train_loader)
    t_start = time.time()
    pbar = tqdm(total=n_samples, desc=f"{method_name} samples")

    while samples_seen < n_samples:
        try:
            data_item = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data_item = next(train_iter)

        batch_n = _batch_size(data_item)

        model.train()
        y_pred, y_true = predict_fn(model, data_item, device)
        y_pred_sig = torch.sigmoid(y_pred)
        loss = loss_fn(y_true, y_pred_sig)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_window.append(loss.item())
        step += 1
        samples_seen += batch_n
        pbar.update(batch_n)

        if time_budget_s is not None and (time.time() - t_start) >= time_budget_s:
            log(
                f"[{method_name}] stopping: time budget {time_budget_s:.0f}s reached "
                f"at step {step}, samples {samples_seen}.",
                filepath=str(log_file),
            )
            break

        if samples_seen - samples_at_last_eval >= eval_every:
            samples_at_last_eval = samples_seen
            train_loss = float(np.mean(train_loss_window)) if train_loss_window else float("nan")
            train_loss_window = []
            val_loss, pearson_R, spearman_R, r2 = evaluate_loader(
                model, valid_loader, predict_fn, loss_fn, device, max_batches=valid_max_batches,
            )
            scheduler.step()
            log(
                f"Samples {samples_seen}/{n_samples} (step {step}): train loss {train_loss:.3f}, "
                f"valid loss {val_loss:.3f}, P.R. {pearson_R:.3f}, S.R. {spearman_R:.3f}, R^2 {r2:.3f}.",
                filepath=str(log_file),
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                log("Model weights successfully saved.", filepath=str(log_file))

    pbar.close()

    model.eval()
    model_save_path = Path(model_save_path)
    if model_save_path.exists():
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    test_loss, pearson_R, spearman_R, r2 = evaluate_loader(
        model, test_loader, predict_fn, loss_fn, device, max_batches=None,
    )
    log(
        f"\n\nTest loss {test_loss:.3f}, P.R. {pearson_R:.3f}, S.R. {spearman_R:.3f}, R^2 {r2:.3f}.",
        filepath=str(log_file),
    )
