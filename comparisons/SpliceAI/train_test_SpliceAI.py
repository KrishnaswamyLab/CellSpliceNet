import argparse
import sys
from pathlib import Path

import torch
from spliceai_pytorch import SpliceAI
from spliceai_pytorch.spliceai_pytorch import SpliceAI_10k

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from setup import COMPARISON_SEQ_LEN, comparison_run_paths, load_splicedata, setup_import_paths, to_coded_seq
from training import add_comparison_args, run_step_training

setup_import_paths()
from log_utils import log
from seed import seed_everything

SPLICEAI_MODEL = "10k"
SPLICEAI_FLANK = 5000
SPLICEAI_CENTER = SpliceAI_10k.S
SPLICEAI_INPUT_LEN = SPLICEAI_FLANK + SPLICEAI_CENTER + SPLICEAI_FLANK


def prepare_spliceai_input(coded_seq: torch.Tensor) -> torch.Tensor:
    """Pad/truncate to SpliceAI-10k layout: 5k flank | 10k center | 5k flank."""
    _, _, seq_len = coded_seq.shape
    if seq_len > SPLICEAI_CENTER:
        start = (seq_len - SPLICEAI_CENTER) // 2
        coded_seq = coded_seq[..., start : start + SPLICEAI_CENTER]
        seq_len = SPLICEAI_CENTER
    pad_center = SPLICEAI_CENTER - seq_len
    pad_left = SPLICEAI_FLANK + pad_center // 2
    pad_right = SPLICEAI_FLANK + (pad_center - pad_center // 2)
    return torch.nn.functional.pad(coded_seq, (pad_left, pad_right))


def predict(model, data_item, device):
    coded_seq = prepare_spliceai_input(to_coded_seq(data_item, device))
    y_true = data_item[2]["psi"].to(device)
    y_pred = model(coded_seq)[:, 0, :]
    return y_pred, y_true


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(description="SpliceAI comparison baseline.")
    add_comparison_args(cmd_parser)
    cmd_args = cmd_parser.parse_known_args()[0]
    seed_everything(cmd_args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splicedata(cmd_args.batch_size, data_tag=cmd_args.data_tag, num_workers=cmd_args.num_workers)

    model = SpliceAI.from_preconfigured(SPLICEAI_MODEL)
    model.conv1 = torch.nn.Conv1d(
        in_channels=2,
        out_channels=model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
    )
    model.conv_last = torch.nn.Conv1d(
        in_channels=model.conv_last.in_channels,
        out_channels=1,
        kernel_size=model.conv_last.kernel_size,
        stride=model.conv_last.stride,
        padding=model.conv_last.padding,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cmd_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_fn = torch.nn.MSELoss()

    log_file, model_save_path = comparison_run_paths("SpliceAI", cmd_args.data_tag, cmd_args.random_seed)

    log(
        f"[SpliceAI] Training begins (window={COMPARISON_SEQ_LEN}, model_input={SPLICEAI_INPUT_LEN}).",
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
        method_name="SpliceAI",
    )
