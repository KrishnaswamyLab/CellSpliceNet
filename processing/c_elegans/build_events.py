#!/usr/bin/env python3
"""
build_events.py — one table of every (event, replicate sample) PSI measurement.

Inputs (<data_root>/c_elegans/raw/):
    full_data_Feb01.tsv                       per-replicate quantification (SE events and the CE / CI
                                              control exons / introns), incl. replicates with 0 reads
    2025-02-01-13-40-05_main_data.csv         event-level sequence lengths (p_seq_len, sj_seq_len)

Output:
    <data_root>/c_elegans/events.csv   event_id, event_type, neuron, replicate, nb_reads, PSI (empty when the
                                       replicate has no reads), intron/exon coordinates, gene_length, gene_id,
                                       p_seq_len, sj_seq_len

Per-neuron targets are made from this table by splits/make_splits.py. The published per-neuron data are
PSI = mean over the replicates with reads and nb_reads = mean over all replicates.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from processing.common import organism_paths  # noqa: E402

RAW, PROCESSED, OUT_DIR = organism_paths("c_elegans")
OUT_FILE = (OUT_DIR.parent if OUT_DIR == PROCESSED else OUT_DIR) / "events.csv"
INT_COLS = ["nb_reads", "intron_start", "intron_end", "exon_start", "exon_end", "gene_length", "p_seq_len", "sj_seq_len"]


def main():
    ev = pd.read_csv(RAW / "full_data_Feb01.tsv", sep="\t",
                     usecols=["event_id", "sample_id", "nb_reads", "PSI", "intron_start", "intron_end",
                              "exon_start", "exon_end", "gene_length", "gene_id"])
    ev = ev.rename(columns={"sample_id": "replicate"})
    ev.insert(1, "event_type", ev.event_id.str.split("_").str[0])
    ev.insert(2, "neuron", ev.replicate.str.replace(r"r\d+$", "", regex=True))
    lens = (pd.read_csv(RAW / "2025-02-01-13-40-05_main_data.csv", usecols=["event_id", "p_seq_len", "sj_seq_len"])
            .drop_duplicates("event_id").set_index("event_id"))
    ev["p_seq_len"] = ev.event_id.map(lens.p_seq_len)
    ev["sj_seq_len"] = ev.event_id.map(lens.sj_seq_len)
    for c in INT_COLS:
        ev[c] = ev[c].round().astype("Int64")
    ev.to_csv(OUT_FILE, index=False)
    print(f"{len(ev):,} replicate rows ({ev.PSI.notna().sum():,} with PSI), {ev.event_id.nunique():,} events, "
          f"{ev.neuron.nunique()} neurons -> {OUT_FILE}")


if __name__ == "__main__":
    main()
