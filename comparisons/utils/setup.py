"""Shared import and path setup for comparison train_test scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISONS_UTILS = REPO_ROOT / "comparisons" / "utils"
SRC_ROOT = REPO_ROOT / "src"

# Dataloader encodings (see src/data/splicedata_dataloader.py).
PAD_INDEX = 0
ANNOTATION_EXON = 1


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


def truncate_sequence_batch(
    sequence: torch.Tensor,
    annotation: torch.Tensor,
    max_len: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop a window of length ``max_len`` centered on the exon of interest.

    The gene sequence is left-aligned, so the exon (``annotation == EXON``) can sit
    anywhere; a naive front-truncation frequently drops it, which silently starves
    every baseline of the signal it needs. Center the window on the exon per sample,
    falling back to the non-pad content center for events with no exon in range. The
    output length matches the previous front-truncation (``min(L, max_len)``).
    """
    seq_len = sequence.shape[-1]
    if seq_len <= max_len:
        return sequence, annotation

    positions = torch.arange(seq_len, dtype=torch.float32)
    exon = annotation == ANNOTATION_EXON
    content = sequence != PAD_INDEX
    exon_count = exon.sum(dim=-1)
    exon_center = (positions * exon).sum(dim=-1) / exon_count.clamp(min=1)
    content_center = (positions * content).sum(dim=-1) / content.sum(dim=-1).clamp(min=1)
    center = torch.where(exon_count > 0, exon_center, content_center)

    start = (center.round().long() - max_len // 2).clamp(0, seq_len - max_len)
    idx = start.unsqueeze(-1) + torch.arange(max_len)
    return torch.gather(sequence, -1, idx), torch.gather(annotation, -1, idx)


def comparison_batch_inputs(data_item, device: torch.device, max_len: int = 4096):
    sequence, annotation = data_item[1]
    sequence, annotation = truncate_sequence_batch(sequence, annotation, max_len=max_len)
    return sequence.to(device), annotation.to(device)


def to_coded_seq(data_item, device: torch.device, max_len: int = 4096) -> torch.Tensor:
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
