#!/usr/bin/env python3
"""
prep_01_sequences.py — Per-event primary sequence encoding (windowed).

For every cassette-exon event listed in events_coordinates.csv, fetch a 25 kb
window of genomic sequence centred on the cassette exon (window edges already
chosen by prep_02 and recorded in `window_start_in_gene`/`window_end_in_gene`).
Sequences shorter than 25 kb (gene shorter than the window) are zero-padded on
the right.

This replaces the original gene-keyed scheme (truncate every gene to the first
25 kb), which is fine for C. elegans (tiny genes) but drops ~80 % of human
events whose cassette exon sits past position 25 000 in the gene.

Inputs:
    preprocessing/processed/events_coordinates.csv (from prep_02)
    data/hg38/gencode.v38.annotation.gtf.gz     (gene chrom/strand/genomic span)
    data/hg38/GRCh38.p14.genome.fa              (genome FASTA, .fai required)

Output:
    preprocessing/processed/encoded_sequences.pth
        { event_id (str) : Tensor[P_WIN_LEN] (long) }
        Encoding: A=1, G=2, T=3, C=4, N/unknown=0, PAD=0
"""

import gzip
import re
from pathlib import Path

import pysam
import pandas as pd
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from processing.common import organism_paths  # noqa: E402

RAW, PROCESSED, OUT_DIR = organism_paths("human")
GTF_FILE     = RAW / "hg38/gencode.v38.annotation.gtf.gz"
FASTA_FILE   = RAW / "hg38/GRCh38.p14.genome.fa"
COORDS_FILE  = OUT_DIR / "events_coordinates.csv"

# ── Config ───────────────────────────────────────────────────────────────────
P_WIN_LEN = 25_000        # must match prep_02.P_WIN_LEN
ENCODING  = {"A": 1, "G": 2, "T": 3, "C": 4}     # unknown/N → 0

# ── Helpers ──────────────────────────────────────────────────────────────────
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def parse_gene_coords(gtf_file: Path, wanted_ids: set) -> dict:
    """Return {gene_id: {chrom, start, end, strand}} for each wanted gene.
    Matches on base Ensembl ID (ignores version suffix)."""
    base_to_full = {gid.split(".")[0]: gid for gid in wanted_ids}
    coords = {}
    with gzip.open(gtf_file, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 9 or cols[2] != "gene":
                continue
            m = re.search(r'gene_id "([^"]+)"', cols[8])
            if m is None:
                continue
            gtf_base = m.group(1).split(".")[0]
            if gtf_base not in base_to_full:
                continue
            coords[base_to_full[gtf_base]] = {
                "chrom":  cols[0],
                "start":  int(cols[3]),   # 1-based
                "end":    int(cols[4]),   # 1-based inclusive
                "strand": cols[6],
            }
            if len(coords) == len(wanted_ids):
                break
    return coords


def fetch_window_seq(fa: pysam.FastaFile, gc: dict,
                     win_start: int, win_end: int) -> str:
    """Fetch gene-relative half-open [win_start, win_end), oriented 5′→3′."""
    if gc["strand"] == "+":
        g0 = gc["start"] - 1 + win_start
        g1 = gc["start"] - 1 + win_end
        return fa.fetch(gc["chrom"], g0, g1)
    else:
        g0 = gc["end"] - win_end
        g1 = gc["end"] - win_start
        return revcomp(fa.fetch(gc["chrom"], g0, g1))


def encode_seq(seq: str, pad_to: int) -> torch.Tensor:
    enc = [ENCODING.get(b.upper(), 0) for b in seq]
    enc += [0] * (pad_to - len(enc))
    return torch.tensor(enc, dtype=torch.long)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Reading events: {COORDS_FILE}")
    events = pd.read_csv(COORDS_FILE)
    needed = {"event_id", "gene_id", "window_start_in_gene", "window_end_in_gene"}
    missing = needed - set(events.columns)
    if missing:
        raise RuntimeError(
            f"events_coordinates.csv is missing {missing}. Run prep_02 first."
        )
    print(f"  {len(events):,} events on {events['gene_id'].nunique():,} genes")

    gene_ids = set(events["gene_id"].unique())
    print("Parsing GTF …")
    gene_coords = parse_gene_coords(GTF_FILE, gene_ids)
    print(f"  found coords for {len(gene_coords)} / {len(gene_ids)} genes")

    print("Fetching + encoding per-event 25 kb windows …")
    fa = pysam.FastaFile(str(FASTA_FILE))
    encoded = {}
    skipped = 0
    for _, row in events.iterrows():
        gid = row["gene_id"]
        eid = row["event_id"]
        if gid not in gene_coords:
            skipped += 1
            continue
        seq = fetch_window_seq(
            fa, gene_coords[gid],
            int(row["window_start_in_gene"]), int(row["window_end_in_gene"]),
        )
        encoded[eid] = encode_seq(seq, P_WIN_LEN)
    fa.close()

    print(f"  encoded {len(encoded):,} events (skipped {skipped})")
    out = OUT_DIR / "encoded_sequences.pth"
    torch.save(encoded, out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
