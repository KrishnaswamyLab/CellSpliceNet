"""Central path resolution for CellSpliceNet.

All repo-relative and data_config.ini path tokens are resolved here.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset"
OUTPUTS_ROOT = REPO_ROOT / "outputs"

HUMAN_CONFIG = DATASET_ROOT / "human" / "data_config.ini"
CELEGANS_CONFIG = DATASET_ROOT / "c_elegans" / "data_config.ini"

FILES_SECTION = "files"

_TOKEN_ROOT = "$ROOT"
_TOKEN_DATASET = "$DATASET"
_TOKEN_CONFIG_DIR = "$CONFIG_DIR"


def _substitute_tokens(value: str, *, config_file: Path | None, dataset_root: Path) -> str:
    config_dir = config_file.parent if config_file is not None else None
    replacements = {
        _TOKEN_ROOT: str(REPO_ROOT),
        _TOKEN_DATASET: str(dataset_root),
    }
    if config_dir is not None:
        replacements[_TOKEN_CONFIG_DIR] = str(config_dir)
    for token, target in replacements.items():
        value = value.replace(token, target)
    return value


def resolve(
    value: str,
    *,
    config_file: Path | None = None,
    dataset_root: Path | None = None,
) -> Path:
    """Resolve a path string from data_config.ini."""
    value = value.strip()
    if not value:
        raise ValueError("empty path")
    dataset_root = dataset_root or DATASET_ROOT
    expanded = _substitute_tokens(value, config_file=config_file, dataset_root=dataset_root)
    path = Path(expanded)
    if path.is_absolute():
        return path.resolve()
    if config_file is not None:
        return (config_file.parent / path).resolve()
    return (REPO_ROOT / path).resolve()


def preset_config_for_tag(tag: str) -> Path:
    """Default data_config.ini: human (GTEx) or c_elegans (worm)."""
    if tag.lower() in ("gtex", "human"):
        return HUMAN_CONFIG
    return CELEGANS_CONFIG


def _read_config_value(config_file: Path, section: str, key: str) -> str | None:
    config = configparser.ConfigParser()
    config.read(config_file)
    if section not in config or key not in config[section]:
        return None
    value = config[section][key].strip()
    return value or None


def read_config_path(config_file: Path, section: str, key: str) -> Path:
    value = _read_config_value(config_file, section, key)
    if value is None:
        raise KeyError(f"{section}.{key} missing in {config_file}")
    return resolve(value, config_file=config_file)


def read_config_path_optional(config_file: Path, section: str, key: str) -> Path | None:
    value = _read_config_value(config_file, section, key)
    if value is None:
        return None
    return resolve(value, config_file=config_file)


@dataclass(frozen=True)
class DatasetPaths:
    config_file: Path
    enc_seq_file: Path
    enc_sj_file: Path
    spliceregion_inds: Path
    train_data_file: Path
    valid_data_file: Path
    test_data_file: Path
    events_coordinates: Path | None
    structure_data_root: Path
    scatter_coeffs_dir: Path
    mean_vec_dir: Path | None
    graph_metric_dir: Path | None

    @property
    def expression_data_root(self) -> Path:
        return self.train_data_file


def load_dataset_bundle(config_file: str | Path) -> DatasetPaths:
    """Load and resolve all paths declared in a dataset's data_config.ini."""
    config_path = resolve(str(config_file))

    enc_seq_file = read_config_path(config_path, FILES_SECTION, "enc_seq_file")
    enc_sj_file = read_config_path(config_path, FILES_SECTION, "enc_sj_file")
    spliceregion_inds = read_config_path(config_path, FILES_SECTION, "spliceregion_inds")
    train_data_file = read_config_path(config_path, FILES_SECTION, "train_data_file")
    valid_data_file = read_config_path(config_path, FILES_SECTION, "valid_data_file")
    test_data_file = read_config_path(config_path, FILES_SECTION, "test_data_file")
    events_coordinates = read_config_path_optional(config_path, FILES_SECTION, "events_coordinates")

    structure_data_root = read_config_path(config_path, FILES_SECTION, "structure_data_root")
    scatter_coeffs_dir = read_config_path(config_path, FILES_SECTION, "scatter_coeffs_dir")

    return DatasetPaths(
        config_file=config_path,
        enc_seq_file=enc_seq_file,
        enc_sj_file=enc_sj_file,
        spliceregion_inds=spliceregion_inds,
        train_data_file=train_data_file,
        valid_data_file=valid_data_file,
        test_data_file=test_data_file,
        events_coordinates=events_coordinates,
        structure_data_root=structure_data_root,
        scatter_coeffs_dir=scatter_coeffs_dir,
        mean_vec_dir=read_config_path_optional(config_path, FILES_SECTION, "mean_vec_dir"),
        graph_metric_dir=read_config_path_optional(config_path, FILES_SECTION, "graph_metric_dir"),
    )


def resolve_training_paths(args) -> Any:
    """Resolve data_config.ini and attach the dataset path bundle onto args."""
    config_file = resolve(args.config_fname)
    args.config_fname = str(config_file)
    args.dataset_paths = load_dataset_bundle(config_file)
    return args


def run_output_dir(model: str, run_key: str, dataset_type: str) -> Path:
    return OUTPUTS_ROOT / model / f"{run_key}{dataset_type}"


def ensure_run_output_layout(output_dir: Path) -> None:
    for sub in (
        "model",
        "scatter_valid",
        "scatter_valid/psi",
        "scatter_valid/delta-psi",
        "codes",
    ):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
