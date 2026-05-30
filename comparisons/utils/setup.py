"""Shared import and path setup for comparison train_test scripts."""
from __future__ import annotations

import sys
from pathlib import Path

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


def load_splicedata(batch_size: int, data_tag: str = "replicate"):
    """Load worm/human splits via data_config.ini (same path flow as train_full.py)."""
    setup_import_paths()
    from args import argparser_fn
    from data.splicedata_dataloader import splicedata_dataloader

    args = argparser_fn(data_tag, batch_size)
    data = splicedata_dataloader(args)
    data.setup()
    data.setup_hparams(args)
    return data
