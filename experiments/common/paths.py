"""Paths shared by the experiment scripts.

Data come from the data root ($CSN_DATA_ROOT, else <repo>/dataset), training runs from
$CSN_RUNS (default <repo>/outputs)/<model>/<key><dataset_type>/ (src/train_full.py), and experiment results go to
$CSN_EXP_OUT (default <repo>/outputs/experiments)/<family>/.
"""
import configparser
import glob
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
for p in (str(REPO), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA = Path(os.environ.get("CSN_DATA_ROOT") or REPO / "dataset")
HUMAN = DATA / "human"
WORM = DATA / "c_elegans"
# Training runs: $CSN_RUNS (e.g. an older outputs folder) else <repo>/outputs.
RUNS = Path(os.environ.get("CSN_RUNS") or REPO / "outputs")

# Old (pre-repo) split names -> data-root split folders.
SPLIT_ALIASES = {
    "human": {"orig": "gene", "orig8": "gene_k8", "chrom8": "chrom_k8", "tissue": "cell", "tissue8": "cell_k8",
              "combo": "gene_x_cell", "combo_em": "gene_x_cell_em", "combo_em_all": "gene_x_cell_em_all"},
    "c_elegans": {"SE": "rows_se", "alltypes": "rows", "eventbased": "event_sample1",
                  "genebased": "gene_sample1", "neuronbased": "cell_sample1"},
}


def out_dir(*parts) -> Path:
    """<$CSN_EXP_OUT or outputs/experiments>/<parts...>, created."""
    p = Path(os.environ.get("CSN_EXP_OUT") or REPO / "outputs/experiments").joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def split_dir(name: str, organism: str = "human") -> Path:
    return DATA / organism / "splits" / SPLIT_ALIASES.get(organism, {}).get(name, name)


def split_csv(name: str, part: str, organism: str = "human") -> Path:
    return split_dir(name, organism) / f"{part}.csv"


def split_config(name: str, organism: str = "human") -> Path:
    return split_dir(name, organism) / "data_config.ini"


def read_split(name: str, part: str, organism: str = "human", **kw):
    import pandas as pd
    return pd.read_csv(split_csv(name, part, organism), **kw)


def scatter_dir(organism: str = "human", name: str = None) -> Path:
    name = name or {"human": "scatter_493", "c_elegans": "scatter_243"}[organism]
    return DATA / organism / "processed/expression" / name


def run_dirs(pattern: str, model: str = "*"):
    """Training run dirs whose dataset tag starts with `pattern` (e.g. 'hg_gene_k8_csn_seqonly')."""
    return sorted(glob.glob(str(RUNS / model / f"*01Feb2025_{pattern}*")))


def split_names(name: str, organism: str = "human"):
    """The data-root name of a split plus its old names (runs trained before the repo carry those)."""
    new = SPLIT_ALIASES.get(organism, {}).get(name, name)
    return [new] + [o for o, n in SPLIT_ALIASES.get(organism, {}).items() if n == new]


def split_runs(prefix: str, split: str, rest: str = "", model: str = "*", organism: str = "human"):
    """Run dirs '<prefix>_<split>_<rest>*' under the split's new or old name, e.g.
    split_runs('hg', 'gene_k8', 'csn_seqonly') also finds the earlier 'hg_orig8_csn_seqonly' runs."""
    out = []
    for n in split_names(split, organism):
        out += run_dirs(f"{prefix}_{n}_{rest}", model)
    return sorted(set(out))


def released_checkpoint() -> Path:
    """The published C. elegans checkpoint (README / Google Drive 'CellSpliceNet.pth')."""
    return Path(os.environ.get("CSN_RELEASED_CKPT") or WORM / "checkpoints/cellsplicenet_released.pth")


def _tokenize(v: str) -> str:
    """Paths inside the data root are written as $DATASET/..., so a derived config moves with the data."""
    if v.startswith("$"):
        return v
    root, p = os.path.realpath(DATA), os.path.realpath(v)
    return "$DATASET/" + os.path.relpath(p, root) if p == root or p.startswith(root + os.sep) else v


def derive_config(base_ini, out_ini, **files) -> Path:
    """Copy a data_config.ini with some [files] entries replaced (absolute paths or $DATASET tokens).
    $CONFIG_DIR entries of the base keep pointing at the base config's folder."""
    base_ini, out_ini = Path(base_ini), Path(out_ini)
    c = configparser.ConfigParser()
    c.read(base_ini)
    for k, v in c["files"].items():
        c["files"][k] = _tokenize(v.replace("$CONFIG_DIR", str(base_ini.parent.resolve())))
    for k, v in files.items():
        c["files"][k] = _tokenize(str(v))
    out_ini.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ini, "w") as f:
        c.write(f)
    return out_ini


def tissue_folder_map():
    """{GTEx export folder prefix: ASCOT tissue name}."""
    from processing.human.expression.tissue_folder_map import FOLDER_TO_ASCOT
    return FOLDER_TO_ASCOT


def gtex_counts_dir() -> Path:
    return HUMAN / "raw/gtex_counts/exported_counts"
