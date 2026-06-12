import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from spliceai_pytorch import SpliceAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from setup import ANNOTATION_EXON, comparison_run_paths, load_splicedata, setup_import_paths, to_coded_seq
from training import add_comparison_args, run_step_training

setup_import_paths()
from log_utils import log
from nn.expression import GraphExpressionModality
from seed import seed_everything

SPLICEAI_MODEL = "10k"
EXPR_HIDDEN = 128
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


def _scatter_feature_dim(scatter_dir: Path) -> int:
    for pattern in ("scatter_coeffs_*.pt", "*.pt"):
        for scatter_path in sorted(scatter_dir.glob(pattern)):
            scatter = torch.load(scatter_path, map_location="cpu", weights_only=False)
            return int(scatter.shape[1] * scatter.shape[2])
    raise FileNotFoundError(f"No scatter tensors under {scatter_dir}")


def build_spliceai_backbone() -> SpliceAI:
    model = SpliceAI.from_preconfigured(SPLICEAI_MODEL)
    model.conv1 = nn.Conv1d(
        in_channels=2,
        out_channels=model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
    )
    model.conv_last = nn.Conv1d(
        in_channels=model.conv_last.in_channels,
        out_channels=1,
        kernel_size=model.conv_last.kernel_size,
        stride=model.conv_last.stride,
        padding=model.conv_last.padding,
    )
    return model


def build_expression_encoder(data) -> GraphExpressionModality:
    paths = data.paths
    feat_dim = _scatter_feature_dim(paths.scatter_coeffs_dir)
    if feat_dim % 11 != 0:
        raise ValueError(f"Expected scatter feature dim divisible by 11, got {feat_dim}")
    return GraphExpressionModality(
        exp_dim=feat_dim // 11,
        coeff_dim=feat_dim,
        hidden_dim=EXPR_HIDDEN,
        gene_embed_bool=data.gene_embed_bool,
        bin_exp=data.bin_exp,
        ntype_feature_bool=data.ntype_feature_bool,
        expression_data_root=str(paths.train_data_file.parent),
        save_output_hook=None,
        scatter_coeffs_dir=str(paths.scatter_coeffs_dir),
    )


class SpliceAIWithExpression(nn.Module):
    """SpliceAI sequence logit + CellSpliceNet scatter encoder, fused with a small MLP."""

    def __init__(self, spliceai: SpliceAI, expression_model: GraphExpressionModality):
        super().__init__()
        self.spliceai = spliceai
        self.expression_model = expression_model
        self.fusion = nn.Sequential(
            nn.Linear(1 + expression_model.hidden_dim, EXPR_HIDDEN),
            nn.ReLU(),
            nn.Linear(EXPR_HIDDEN, 1),
        )

    def _sequence_logit(self, coded_seq: torch.Tensor) -> torch.Tensor:
        padded = prepare_spliceai_input(coded_seq)
        out = self.spliceai(padded)[..., 0]
        annotation = padded[:, 1, SPLICEAI_PRED_START : SPLICEAI_PRED_START + SPLICEAI_PRED_LEN]
        exon_mask = (annotation == ANNOTATION_EXON).type_as(out)
        denom = exon_mask.sum(dim=1, keepdim=True)
        seq_logit = (out * exon_mask).sum(dim=1, keepdim=True) / denom.clamp(min=1.0)
        no_exon = denom.squeeze(1) == 0
        if no_exon.any():
            seq_logit = seq_logit.clone()
            seq_logit[no_exon] = out[no_exon].mean(dim=1, keepdim=True)
        return seq_logit

    def forward(self, coded_seq: torch.Tensor, metadata) -> torch.Tensor:
        seq_logit = self._sequence_logit(coded_seq)
        exp_tokens, _ = self.expression_model.prep_expression(metadata)
        exp_pool, _ = self.expression_model.exp_glob_attn_op(exp_tokens, output_glob_attn=True)
        return self.fusion(torch.cat([seq_logit, exp_pool], dim=-1))


def predict(model, data_item, device):
    coded_seq = to_coded_seq(data_item, device)
    y_pred = model(coded_seq, data_item[0])
    y_true = data_item[2]["psi"].to(device)
    return y_pred, y_true


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(description="SpliceAI + expression comparison baseline.")
    add_comparison_args(cmd_parser)
    cmd_args = cmd_parser.parse_known_args()[0]
    seed_everything(cmd_args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splicedata(cmd_args.batch_size, data_tag=cmd_args.data_tag, num_workers=cmd_args.num_workers)

    model = SpliceAIWithExpression(
        build_spliceai_backbone(),
        build_expression_encoder(data),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cmd_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_fn = torch.nn.MSELoss()

    log_file, model_save_path = comparison_run_paths("SpliceAI_expression", cmd_args.data_tag, cmd_args.random_seed)

    log(
        f"[SpliceAI_expression] Training begins.",
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
        method_name="SpliceAI_expr",
    )
