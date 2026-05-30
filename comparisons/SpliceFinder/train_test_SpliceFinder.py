import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from setup import COMPARISON_SEQ_LEN, comparison_run_paths, load_splicedata, setup_import_paths, to_coded_seq
from training import add_comparison_args, run_step_training

setup_import_paths()
from log_utils import log
from seed import seed_everything

from splicefinder_model import SpliceFinder


def predict(model, data_item, device):
    coded_seq = to_coded_seq(data_item, device)
    y_true = data_item[2]["psi"].to(device)
    y_pred = model(coded_seq)
    return y_pred, y_true


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(description="SpliceFinder comparison baseline.")
    add_comparison_args(cmd_parser)
    cmd_args = cmd_parser.parse_known_args()[0]
    seed_everything(cmd_args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splicedata(cmd_args.batch_size, data_tag=cmd_args.data_tag, num_workers=cmd_args.num_workers)

    model = SpliceFinder(in_channels=2, seq_len=COMPARISON_SEQ_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cmd_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_fn = torch.nn.MSELoss()

    log_file, model_save_path = comparison_run_paths("SpliceFinder", cmd_args.data_tag, cmd_args.random_seed)

    log(f"[SpliceFinder] Training begins (seq_len={COMPARISON_SEQ_LEN}).", filepath=str(log_file))
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
        method_name="SpliceFinder",
    )
