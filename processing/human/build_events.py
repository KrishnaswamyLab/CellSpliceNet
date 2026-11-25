#!/usr/bin/env python3
"""
build_events.py — one table of every (cassette exon, tissue) PSI measurement (step 3).

Inputs:
    <data_root>/human/raw/ascot/gtex_psi.csv.gz   ASCOT wide PSI table (one row per exon, one column per
                                                  tissue, values 0-100, -1 = missing)
    <data_root>/human/processed/events_coordinates.csv   from coords_and_junctions.py
    expression/tissue_folder_map.py               the 49 GTEx tissues with expression scatter coefficients

Output:
    <data_root>/human/events.csv   gene_id, event_type, event_id, PSI, neuron (= tissue), nb_reads,
                                   p_seq_len, sj_seq_len, exon_start, exon_end, intron_start, intron_end,
                                   gene_length, chrom
Splits are made from this table by splits/make_splits.py.

* PSI is divided by 100 so it lies in [0, 1]; PSI == -1 rows are dropped.
* nb_reads is a constant placeholder (1): ASCOT has no read counts, and the model only uses the field
  with --nb_norm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from processing.common import organism_paths  # noqa: E402
from processing.human.expression.tissue_folder_map import FOLDER_TO_ASCOT  # noqa: E402

RAW, PROCESSED, OUT_DIR = organism_paths("human")
PSI_FILE = RAW / "ascot/gtex_psi.csv.gz"
EVENTS_COORDS_FILE = OUT_DIR / "events_coordinates.csv"
# default output is <data_root>/human/events.csv; with --out_dir it goes there
OUT_FILE = (OUT_DIR.parent if OUT_DIR == PROCESSED else OUT_DIR) / "events.csv"
NB_READS_PLACEHOLDER = 1
COLUMNS = ["gene_id", "event_type", "event_id", "PSI", "neuron", "nb_reads", "p_seq_len", "sj_seq_len",
           "exon_start", "exon_end", "intron_start", "intron_end", "gene_length", "chrom"]


def main() -> None:
    coords = pd.read_csv(EVENTS_COORDS_FILE)
    print(f"[1/4] {len(coords):,} encoded cassette-exon events ({coords['gene_id'].nunique():,} genes)")

    psi = pd.read_csv(PSI_FILE).rename(columns={"exon_id": "event_id"})
    psi = psi[psi["event_id"].isin(coords["event_id"])].reset_index(drop=True)
    chrom = psi.set_index("event_id")["exon_location"].str.split(":").str[0]

    tissues = sorted(set(FOLDER_TO_ASCOT.values()))
    present = [t for t in tissues if t in psi.columns]
    missing = [t for t in tissues if t not in psi.columns]
    if missing:
        print(f"      WARNING: {len(missing)} tissues in map but not in PSI file: {missing}")
    print(f"[2/4] {len(psi):,} ASCOT exons, {len(present)} tissues")

    long_df = psi[["event_id"] + present].melt(id_vars="event_id", var_name="neuron", value_name="PSI")
    n_before = len(long_df)
    long_df = long_df[long_df["PSI"] != -1].reset_index(drop=True)
    long_df["PSI"] = (long_df["PSI"].astype(float) / 100.0).clip(lower=0.0, upper=1.0)
    print(f"[3/4] {n_before:,} -> {len(long_df):,} rows after dropping PSI == -1")

    merged = long_df.merge(coords, on="event_id", how="left")
    if merged["gene_id"].isna().any():
        raise RuntimeError(f"{merged['gene_id'].isna().sum()} rows missing gene_id: events_coordinates is incomplete")
    merged["nb_reads"] = NB_READS_PLACEHOLDER
    merged["event_type"] = "SE"
    merged["chrom"] = merged["event_id"].map(chrom)
    merged = merged[COLUMNS]
    merged.to_csv(OUT_FILE, index=False)
    print(f"[4/4] {len(merged):,} rows, {merged['event_id'].nunique():,} events, {merged['gene_id'].nunique():,} genes, "
          f"{merged['neuron'].nunique()} tissues -> {OUT_FILE}")


if __name__ == "__main__":
    main()
