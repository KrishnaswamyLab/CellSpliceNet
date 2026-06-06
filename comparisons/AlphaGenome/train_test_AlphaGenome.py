import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from setup import add_use_pretrained_arg, comparison_batch_inputs, comparison_run_paths, load_splicedata, setup_import_paths
from training import add_comparison_args, run_step_training

setup_import_paths()
from log_utils import log
from seed import seed_everything

from alphagenome_model import SEQUENCE_LENGTH, AlphaGenomeForPSI


def predict(model, data_item, device):
    sequence, annotation = comparison_batch_inputs(data_item, device, max_len=SEQUENCE_LENGTH)
    y_true = data_item[2]["psi"].to(device)
    logits = model(sequence=sequence, annotation=annotation)["logits"]
    return logits, y_true


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(description="AlphaGenome comparison baseline.")
    add_comparison_args(cmd_parser)
    add_use_pretrained_arg(cmd_parser)
    cmd_args = cmd_parser.parse_known_args()[0]
    seed_everything(cmd_args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splicedata(cmd_args.batch_size, data_tag=cmd_args.data_tag, num_workers=cmd_args.num_workers)

    model = AlphaGenomeForPSI(use_pretrained=cmd_args.use_pretrained).to(device)
    if cmd_args.use_pretrained:
        optimizer = torch.optim.AdamW(model.regression_head.parameters(), lr=cmd_args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(
            list(model.annotation_embedding.parameters())
            + list(model.annotation_embedding_128bp.parameters())
            + list(model.regression_head.parameters()),
            lr=cmd_args.learning_rate,
        )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_fn = torch.nn.MSELoss()

    log_file, model_save_path = comparison_run_paths("AlphaGenome", cmd_args.data_tag, cmd_args.random_seed)

    log(
        f"[AlphaGenome] Training begins (use_pretrained={cmd_args.use_pretrained}).",
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
        method_name="AlphaGenome",
    )
