"""Paths shared by the processing scripts.

Data root: --data_root, else $CSN_DATA_ROOT, else <repo>/dataset. Inputs are read from
<data_root>/<organism>/{raw,processed}; outputs go to <data_root>/<organism>/processed unless --out_dir
is given (useful for test runs, which then also read earlier pipeline outputs from --out_dir).
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def default_data_root() -> Path:
    return Path(os.environ.get("CSN_DATA_ROOT") or REPO_ROOT / "dataset")


def organism_paths(organism: str):
    """Return (raw_dir, processed_dir, out_dir) for 'c_elegans' or 'human'."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--out_dir", default=None)
    a, _ = ap.parse_known_args()
    org = Path(a.data_root) / organism
    out = Path(a.out_dir) if a.out_dir else org / "processed"
    out.mkdir(parents=True, exist_ok=True)
    return org / "raw", org / "processed", out


def add_location_args(parser):
    """--organism / --data-root / --work-dir for the structure pipeline (processing/structure)."""
    parser.add_argument("--organism", choices=["c_elegans", "human"], required=True)
    parser.add_argument("--data-root", dest="data_root", default=str(default_data_root()))
    parser.add_argument("--work-dir", dest="work_dir", default=None,
                        help="folder for processed_sequences.csv, structure_dict.pkl and the scattering moments "
                             "(default: <data_root>/<organism>/processed/structure/work)")


def work_dir_for(args) -> str:
    if args.work_dir:
        return args.work_dir
    return str(Path(args.data_root) / args.organism / "processed" / "structure" / "work")
