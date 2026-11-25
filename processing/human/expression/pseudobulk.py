#!/usr/bin/env python3
"""Tissue pseudobulk of all genes: per-sample log1p(CPM), averaged over the samples of each tissue.

Input:   <data_root>/human/raw/gtex_counts/exported_counts/<tissue>--GRCh38--gencode29/geneCounts.tsv.gz
Output:  <data_root>/human/processed/expression/tissue_pseudobulk.csv.gz  (genes x 49 tissues)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from processing.common import organism_paths  # noqa: E402
from processing.human.expression.tissue_folder_map import FOLDER_TO_ASCOT  # noqa: E402
from processing.human.junctions import counts_dir, tissue_folders  # noqa: E402


def main():
    raw, processed, out = organism_paths("human")
    cols = {}
    for f in tissue_folders(raw):
        g = pd.read_csv(counts_dir(raw) / f / "geneCounts.tsv.gz", sep="\t", index_col=0)
        cols[FOLDER_TO_ASCOT[f.split("--")[0]]] = np.log1p(g / g.sum(0) * 1e6).mean(1)
        print("  pseudobulk", f, flush=True)
    (out / "expression").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols)[sorted(cols)].to_csv(out / "expression/tissue_pseudobulk.csv.gz")


if __name__ == "__main__":
    main()
