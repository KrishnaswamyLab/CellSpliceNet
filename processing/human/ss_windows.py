#!/usr/bin/env python3
"""Single-nucleotide windows around the four splice sites of every cassette exon.

ASCOT coordinates (1-based): exon s..e, skipping intron b1..b2 (exon_boundary = the skipping junction).
Four 350-nt segments, genomic left to right:
  A [b1-100, b1+249]  upstream-exon donor + intron start      B [s-300, s+49]  intron end + exon start
  C [e-49,  e+300]    exon end + intron start                  D [b2-249, b2+100] intron end + downstream acceptor
On the '-' strand the concatenation is reverse-complemented, so segments are always in transcript order
(upstream donor, exon acceptor, exon donor, downstream acceptor). Encoding A,C,G,T,N = 0..4; N-padded
at chromosome ends.

Inputs:  <data_root>/human/events.csv, raw/ascot/gtex_psi.csv.gz, raw/hg38/GRCh38.p14.genome.fa
Output:  <data_root>/human/processed/ss_windows.npz  (event_id, seq uint8 [n_events, 1400])
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from processing.common import organism_paths  # noqa: E402
from processing.human.genome import Genome, parse_interval  # noqa: E402

COMP = str.maketrans("ACGTN", "TGCAN")
LUT = np.full(256, 4, np.uint8)
for i, c in enumerate("ACGT"):
    LUT[ord(c)] = i


def main():
    raw, processed, out = organism_paths("human")
    events = pd.read_csv(processed.parent / "events.csv", usecols=["event_id"]).event_id.unique()
    asc = pd.read_csv(raw / "ascot/gtex_psi.csv.gz", encoding="utf-8-sig",
                      usecols=["exon_id", "exon_strand", "exon_location", "exon_boundary"]).set_index("exon_id").loc[events]
    g = Genome(raw / "hg38/GRCh38.p14.genome.fa")

    def get(ch, a, b):  # 1-based inclusive, N-padded
        n = g.length(ch)
        return "N" * max(0, 1 - a) + g.fetch(ch, max(a, 1) - 1, min(b, n)) + "N" * max(0, b - n)

    seqs = np.zeros((len(asc), 1400), np.uint8)
    for i, (e, r) in enumerate(asc.iterrows()):
        ch, s, en = parse_interval(r.exon_location)
        _, b1, b2 = parse_interval(r.exon_boundary)
        seq = get(ch, b1 - 100, b1 + 249) + get(ch, s - 300, s + 49) + get(ch, en - 49, en + 300) + get(ch, b2 - 249, b2 + 100)
        if r.exon_strand == "-":
            seq = seq.translate(COMP)[::-1]
        seqs[i] = LUT[np.frombuffer(seq.encode(), np.uint8)]
    ag = np.mean((seqs[:, 350 + 298] == 0) & (seqs[:, 350 + 299] == 2))
    gt = np.mean((seqs[:, 700 + 50] == 2) & (seqs[:, 700 + 51] == 3))
    print(f"events {len(seqs):,}  AG before exon {ag:.3f}  GT after exon {gt:.3f}", flush=True)
    np.savez_compressed(out / "ss_windows.npz", event_id=np.array(asc.index), seq=seqs)


if __name__ == "__main__":
    main()
