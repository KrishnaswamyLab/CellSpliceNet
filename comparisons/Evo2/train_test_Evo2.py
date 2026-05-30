import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from setup import comparison_batch_inputs, comparison_run_paths, load_splicedata, setup_import_paths
from training import add_comparison_args, run_step_training

setup_import_paths()
from log_utils import log
from seed import seed_everything

from evo2_model import Evo2ForPSI, DEFAULT_MAX_LENGTH


def predict(model, data_item, device):
    sequence, annotation = comparison_batch_inputs(data_item, device)
    y_true = data_item[2]["psi"].to(device)
    y_pred = model(sequence=sequence, annotation=annotation)
    return y_pred, y_true


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(description="Evo2 comparison baseline (linear probing).")
    add_comparison_args(cmd_parser)
    cmd_args = cmd_parser.parse_known_args()[0]
    seed_everything(cmd_args.random_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Evo2 baseline requires CUDA. Submit via bash/comparison_Evo2.sh on a GPU node.")

    device = torch.device("cuda")
    data = load_splicedata(cmd_args.batch_size, data_tag=cmd_args.data_tag, num_workers=cmd_args.num_workers)

    model = Evo2ForPSI(max_length=DEFAULT_MAX_LENGTH).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cmd_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_fn = torch.nn.MSELoss()

    log_file, model_save_path = comparison_run_paths("Evo2", cmd_args.data_tag, cmd_args.random_seed)

    log(
        f"[Evo2] Training begins (seq_len={DEFAULT_MAX_LENGTH}, frozen evo2_7b_base + linear head).",
        filepath=str(log_file),
    )
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
        n_samples=cmd_args.n_samples,
        eval_every=cmd_args.eval_every,
        val_max_batches=cmd_args.val_max_batches,
        time_budget_s=cmd_args.time_budget_s,
        method_name="Evo2",
    )
