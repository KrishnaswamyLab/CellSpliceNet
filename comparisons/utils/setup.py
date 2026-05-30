"""Shared import and path setup for comparison train_test scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISONS_UTILS = REPO_ROOT / "comparisons" / "utils"
SRC_ROOT = REPO_ROOT / "src"


def setup_import_paths() -> None:
    for path in (COMPARISONS_UTILS, SRC_ROOT):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def comparison_results_dir(model_name: str) -> Path:
    out = REPO_ROOT / "comparisons" / "results" / model_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def comparison_run_paths(model_name: str, data_tag: str, random_seed: int) -> tuple[Path, Path]:
    """Log and checkpoint paths keyed by data_tag so replicate/gtex runs do not collide."""
    out = comparison_results_dir(model_name)
    tag = data_tag.replace("/", "_")
    log_file = out / f"log_{tag}_seed-{random_seed}.txt"
    model_save_path = out / f"model_{tag}_seed-{random_seed}.pt"
    return log_file, model_save_path


COMPARISON_SEQ_LEN = 4096


def comparison_seq_len(_data_tag: str = "replicate") -> int:
    """Shared sequence window for comparison baselines (fits ~80GB GPU at batch 64)."""
    return COMPARISON_SEQ_LEN


def truncate_sequence_batch(
    sequence: torch.Tensor,
    annotation: torch.Tensor,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max_len or COMPARISON_SEQ_LEN
    return sequence[..., :max_len], annotation[..., :max_len]


def comparison_batch_inputs(data_item, device: torch.device, max_len: int | None = None):
    sequence, annotation = data_item[1]
    sequence, annotation = truncate_sequence_batch(sequence, annotation, max_len=max_len)
    return sequence.to(device), annotation.to(device)


def to_coded_seq(data_item, device: torch.device, max_len: int | None = None) -> torch.Tensor:
    sequence, annotation = comparison_batch_inputs(data_item, device, max_len=max_len)
    return torch.hstack((sequence[:, None, :], annotation[:, None, :])).float()


def load_splicedata(batch_size: int, data_tag: str = "replicate", num_workers: int = 4):
    """Load worm/human splits via data_config.ini (same path flow as train_full.py)."""
    setup_import_paths()
    from args import argparser_fn
    from data.splicedata_dataloader import splicedata_dataloader

    args = argparser_fn(data_tag, batch_size)
    args.num_workers = num_workers
    data = splicedata_dataloader(args)
    data.setup()
    data.setup_hparams(args)
    return data
