import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from setup import comparison_results_dir, load_splicedata, setup_import_paths
from training import add_comparison_args, resolve_n_steps, run_step_training

setup_import_paths()
from log_utils import log
from seed import seed_everything

from pangolin_model import Pangolin


def predict(model, data_item, device):
    sequence, annotation = data_item[1]
    coded_seq = torch.hstack((sequence[:, None, :], annotation[:, None, :])).float().to(device)
    y_true = data_item[2]["psi"].to(device)
    y_pred = model(coded_seq)[:, :, 0]
    return y_pred, y_true


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(description="Pangolin comparison baseline.")
    add_comparison_args(cmd_parser)
    cmd_args = cmd_parser.parse_known_args()[0]
    seed_everything(cmd_args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splicedata(cmd_args.batch_size, data_tag=cmd_args.data_tag)
    n_steps = resolve_n_steps(cmd_args.data_tag, cmd_args.n_steps)

    model = Pangolin(in_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cmd_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_fn = torch.nn.MSELoss()

    results_dir = comparison_results_dir("Pangolin")
    log_file = results_dir / f"log_seed-{cmd_args.random_seed}.txt"
    model_save_path = results_dir / f"model_seed-{cmd_args.random_seed}.pt"

    log("[Pangolin] Training begins.", filepath=str(log_file))
    run_step_training(
        model=model,
        data=data,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        predict_fn=predict,
        log_file=log_file,
        model_save_path=model_save_path,
        n_steps=n_steps,
        eval_every=cmd_args.eval_every,
        valid_max_batches=cmd_args.valid_max_batches,
        time_budget_s=cmd_args.time_budget_s,
        method_name="Pangolin",
    )
