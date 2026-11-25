"""Inputs, outputs and metrics shared by the human generalization experiments.

Data products (built by processing/human, see processing/README.md) live in <data_root>/human/processed;
results go to <$CSN_EXP_OUT or outputs/experiments>/human_gen/<analysis>/.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.common.paths import (HUMAN, RUNS, out_dir, read_split, scatter_dir, split_dir, split_names,  # noqa: F401
                                      split_runs, tissue_folder_map, gtex_counts_dir)

PROCESSED = HUMAN / "processed"
EVENT_FEATURES = PROCESSED / "event_features.csv"
EVENT_VARIABILITY = PROCESSED / "event_variability.csv"
DONOR = PROCESSED / "donor"
SS_WINDOWS = PROCESSED / "ss_windows.npz"
JUNCTION_COUNTS = PROCESSED / "junction_counts.csv.gz"
PSEUDOBULK = PROCESSED / "expression/tissue_pseudobulk.csv.gz"
SCATTER = scatter_dir("human")                 # scatter_coeffs_<tissue>.pt, moments__<folder>.csv
SF_ORDER = SCATTER / "exact_order_of_splice_factors.csv"
ASCOT_PSI = HUMAN / "raw/ascot/gtex_psi.csv.gz"
ASCOT_NAUC = HUMAN / "raw/ascot/gtex_nauc.csv.gz"
GENOME = HUMAN / "raw/hg38/GRCh38.p14.genome.fa"
PARTS = ("train", "valid", "test")

# the 10 sequence features of the feature baselines (processed/event_features.csv)
FC = ["log_exon", "log_up", "log_dn", "len_mod3", "gc_exon", "gc_up", "gc_dn", "maxent3", "maxent5", "ag_up"]


def out(name):
    return out_dir("human_gen", name)


def all_rows(cols=None):
    """Every (exon, tissue) row: train + valid + test of the gene split."""
    return pd.concat([read_split("gene", k, usecols=cols) for k in PARTS], ignore_index=True)


def baselines_file(split):
    """Test-set baseline predictions of a split (split_baselines.py; cell: tissue_baselines.py)."""
    name = split_names(split)[0]
    return out("baselines") / ("cell_test_baselines.csv" if name == "cell" else f"{name}_test_feature_baselines.csv")


def within(d, c, min_cells=5):
    """Within-event Spearman: correlation of per-event mean-centred targets and predictions, events
    measured in >= min_cells cells."""
    d = d[d.groupby("event_id").neuron.transform("count") >= min_cells]
    if d.empty:
        return np.nan
    a = d.PSI - d.groupby("event_id").PSI.transform("mean")
    b = d[c] - d.groupby("event_id")[c].transform("mean")
    return spearmanr(a, b).correlation if b.std() > 1e-9 else 0.0
