import argparse
import sys
from pathlib import Path

import torch
from spliceai_pytorch import SpliceAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from setup import ANNOTATION_EXON, comparison_run_paths, load_splicedata, setup_import_paths, to_coded_seq
from training import add_comparison_args, run_step_training

setup_import_paths()
from log_utils import log
from seed import seed_everything

SPLICEAI_MODEL = "10k"
# SpliceAI_10k.forward hardcodes `return x[..., 5000:5000 + 5000]`: it only emits
# predictions for input positions [5000:10000], a fixed 5000-wide band.
SPLICEAI_PRED_START = 5000
SPLICEAI_PRED_LEN = 5000
SPLICEAI_INPUT_LEN = SPLICEAI_PRED_START + SPLICEAI_PRED_LEN  # 10000


def prepare_spliceai_input(coded_seq: torch.Tensor) -> torch.Tensor:
    """Center the (exon-centered) window inside SpliceAI's [5000:10000] output band.

    The model only predicts input positions 5000-9999, so the real sequence must
    land inside that band. Place the window center at input 7500 and pad to the
    10000-long input the model expects; for the 4096 window this keeps the whole
    sequence within the predicted band.
    """
    _, _, w = coded_seq.shape
    pred_center = SPLICEAI_PRED_START + SPLICEAI_PRED_LEN // 2  # 7500
    pad_left = min(max(pred_center - w // 2, 0), max(SPLICEAI_INPUT_LEN - w, 0))
    pad_right = max(SPLICEAI_INPUT_LEN - w - pad_left, 0)
    return torch.nn.functional.pad(coded_seq, (pad_left, pad_right))


def predict(model, data_item, device):
    coded_seq = to_coded_seq(data_item, device)  # [B, 2, W], exon-centered
    padded = prepare_spliceai_input(coded_seq)    # [B, 2, 10000]
    out = model(padded)[..., 0]                   # [B, 5000] over input[5000:10000]

    annotation = padded[:, 1, SPLICEAI_PRED_START : SPLICEAI_PRED_START + SPLICEAI_PRED_LEN]
    exon_mask = (annotation == ANNOTATION_EXON).type_as(out)  # [B, 5000]
    denom = exon_mask.sum(dim=1, keepdim=True)
    y_pred = (out * exon_mask).sum(dim=1, keepdim=True) / denom.clamp(min=1.0)
    no_exon = denom.squeeze(1) == 0
    if no_exon.any():
        y_pred = y_pred.clone()
        y_pred[no_exon] = out[no_exon].mean(dim=1, keepdim=True)

    y_true = data_item[2]["psi"].to(device)
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
        f"[SpliceAI] Training begins.",
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
