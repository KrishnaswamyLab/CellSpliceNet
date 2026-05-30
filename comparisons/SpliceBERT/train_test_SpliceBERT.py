import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from setup import COMPARISON_SEQ_LEN, comparison_batch_inputs, comparison_run_paths, load_splicedata, setup_import_paths
from training import add_comparison_args, resolve_n_samples, run_step_training

setup_import_paths()
from log_utils import log
from seed import seed_everything

from splicebert_model import SpliceBert

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def predict(model, data_item, device):
    sequence, annotation = comparison_batch_inputs(data_item, device)
    y_true = data_item[2]["psi"].to(device)
    y_pred = model(sequence=sequence, annotation=annotation)
    return y_pred, y_true


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(description="SpliceBERT comparison baseline.")
    add_comparison_args(cmd_parser)
    cmd_parser.set_defaults(learning_rate=1e-4)
    cmd_args = cmd_parser.parse_known_args()[0]
    seed_everything(cmd_args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splicedata(cmd_args.batch_size, data_tag=cmd_args.data_tag, num_workers=cmd_args.num_workers)
    n_samples = resolve_n_samples(cmd_args.data_tag, cmd_args.n_samples)

    model = SpliceBert(device=device, data_tag=cmd_args.data_tag).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cmd_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_fn = torch.nn.MSELoss()

    log_file, model_save_path = comparison_run_paths("SpliceBERT", cmd_args.data_tag, cmd_args.random_seed)

    log(
        f"[SpliceBERT] Training begins (input truncate {COMPARISON_SEQ_LEN}, "
        f"model max {model.max_seq_len}).",
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
        n_samples=n_samples,
        eval_every=cmd_args.eval_every,
        valid_max_batches=cmd_args.valid_max_batches,
        time_budget_s=cmd_args.time_budget_s,
        method_name="SpliceBERT",
    )
