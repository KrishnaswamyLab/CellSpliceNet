#!/usr/bin/env python3
"""
coords_and_junctions.py  —  Local splice-junction encoding + coord conversion (step 1)

Processes cassette exon (SE) events only.  For each event it:
  1. Converts genomic → gene-relative coordinates (0-based, 5′→3′ of gene).
  2. Defines a 500 bp window centred on the cassette exon.
  3. Extracts the genomic sequence for that window and encodes it.
  4. Builds a binary annotation vector (exon=1, everything else=0).

Inputs:
    gtex_psi.csv.gz                  – ASCOT PSI table (1-based inclusive coords)
    gencode.v38.annotation.gtf.gz    – gene coordinates
    GRCh38.p14.genome.fa             – genome FASTA (needs .fai index beside it)

Outputs (in <data_root>/human/processed/):
    enc_sj_data.pth        – { event_id : Tensor[2, 500] }
                               row 0 = nucleotide codes (A=1,G=2,T=3,C=4,N/pad=0)
                               row 1 = annotation      (exon=1, other=0)
    spliceregion_inds.csv  – event_id, sj_st_indx, sj_end_indx
                               (0-based, primary-window-relative start/end of
                               the 500 bp SJ window)
    events_coordinates.csv – event_id, gene_id, exon_start, exon_end,
                               intron_start, intron_end, gene_length, p_seq_len,
                               sj_seq_len, window_start_in_gene, window_end_in_gene
                               (exon/intron coords are 0-based, primary-window-
                               relative; window_*_in_gene is gene-relative and
                               feeds sequences.py)

Coordinate conventions
──────────────────────
ASCOT exon_location  "chr19:58352098-58352184"  →  1-based inclusive genomic
ASCOT exon_boundary  "chr19:58351688-58352282"  →  1-based inclusive genomic
                      (full intron-pair span: upstream-intron-start … downstream-intron-end)
GTF                  1-based inclusive

Gene-relative (0-based, always 5′→3′ of the gene):
  + strand:  rel = genomic_1based - gene_start
  − strand:  rel = gene_end       - genomic_1based
  (gene_start and gene_end are GTF 1-based values)
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
GTF_FILE   = RAW / "hg38/gencode.v38.annotation.gtf.gz"
FASTA_FILE = RAW / "hg38/GRCh38.p14.genome.fa"
PSI_FILE   = RAW / "ascot/gtex_psi.csv.gz"

# ── Config ───────────────────────────────────────────────────────────────────
SJ_WIN     = 500          # bp; local splice-junction window (centred on exon)
P_WIN_LEN  = 25_000       # bp; primary sequence window (centred on exon).
                          # Genes shorter than this get a smaller window;
                          # genes longer get a window shifted to fit inside the
                          # gene. All downstream coords are window-relative.
ENCODING   = {"A": 1, "G": 2, "T": 3, "C": 4}   # unknown/N → 0


# ── Helpers ──────────────────────────────────────────────────────────────────
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def parse_gene_coords(gtf_file: Path, wanted_ids: set) -> dict:
    """
    Return {gene_id: {chrom, start, end, strand, length}} for each wanted gene.
    Matches on base Ensembl ID (ignores version suffix) so ASCOT (.11) and
    GENCODE v38 GTF (.12) gene IDs resolve correctly.  Keys in the returned
    dict use the original ASCOT gene_id strings.
    """
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
            orig_gid = base_to_full[gtf_base]
            s, e = int(cols[3]), int(cols[4])   # GTF 1-based inclusive
            coords[orig_gid] = {
                "chrom":  cols[0],
                "start":  s,
                "end":    e,
                "strand": cols[6],
                "length": e - s + 1,
            }
            if len(coords) == len(wanted_ids):
                break
    return coords


def to_gene_rel(genomic_1based: int, gc: dict) -> int:
    """
    Convert a 1-based inclusive genomic position to a 0-based gene-relative
    position (always measured 5′→3′ of the gene).

    + strand:  rel = genomic - gene_start
    − strand:  rel = gene_end - genomic
    """
    if gc["strand"] == "+":
        return genomic_1based - gc["start"]
    else:
        return gc["end"] - genomic_1based


def parse_coord(s: str):
    """Parse 'chrX:start-end' (1-based inclusive) → (chrom, start_int, end_int)."""
    chrom, rest = s.split(":")
    start, end = rest.split("-")
    return chrom, int(start), int(end)


def fetch_window_seq(fa: pysam.FastaFile, gc: dict,
                     win_start: int, win_end: int) -> str:
    """
    Extract win_end - win_start bases from the genome corresponding to
    gene-relative positions [win_start, win_end) (0-based, gene-relative).
    Returns a string already oriented 5′→3′ of the gene.
    """
    if gc["strand"] == "+":
        # gene-relative [ws, we) → genomic 0-based [gene_start-1+ws, gene_start-1+we)
        g0 = gc["start"] - 1 + win_start
        g1 = gc["start"] - 1 + win_end
        return fa.fetch(gc["chrom"], g0, g1)
    else:
        # gene-relative [ws, we) on − strand:
        #   position ws  → genomic 1-based gene_end - ws  → 0-based gene_end - ws - 1
        #   position we-1→ genomic 1-based gene_end-(we-1)→ 0-based gene_end - we
        # pysam range (0-based half-open): [gene_end - we, gene_end - ws)
        g0 = gc["end"] - win_end
        g1 = gc["end"] - win_start
        seq = fa.fetch(gc["chrom"], g0, g1)
        return revcomp(seq)          # orient 5′→3′ of the gene


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    half = SJ_WIN // 2   # 250

    # 1. Load PSI table — cassette exons only, skip multi-exon events (| in location)
    print("Loading PSI data (cassette exons only) …")
    psi = pd.read_csv(PSI_FILE)
    psi = psi[psi["cassette_exon"] == "Yes"].copy()
    psi = psi[~psi["exon_location"].str.contains(r"\|", na=False)].copy()
    psi = psi[~psi["exon_boundary"].str.contains(r"\|", na=False)].copy()
    psi.reset_index(drop=True, inplace=True)
    print(f"  {len(psi)} events after filtering")

    # 2. Parse GTF for gene coordinates
    gene_ids = set(psi["gene_id"].dropna().unique())
    print("Parsing GTF …")
    gene_coords = parse_gene_coords(GTF_FILE, gene_ids)
    print(f"  Found coords for {len(gene_coords)} / {len(gene_ids)} genes")

    # 3. Process each event
    print("Encoding splice junctions …")
    fa = pysam.FastaFile(str(FASTA_FILE))

    sj_dict   = {}          # event_id → Tensor[2, 500]
    sj_rows   = []          # → spliceregion_inds.csv
    coord_rows = []         # → events_coordinates.csv
    skipped   = 0

    for _, row in psi.iterrows():
        gid = row["gene_id"]
        eid = row["exon_id"]

        if gid not in gene_coords:
            skipped += 1
            continue

        gc = gene_coords[gid]

        # ── Exon coordinates (gene-relative, 0-based inclusive) ──────────────
        _, ex_g_start, ex_g_end = parse_coord(row["exon_location"])
        ex_rel_a = to_gene_rel(ex_g_start, gc)
        ex_rel_b = to_gene_rel(ex_g_end,   gc)
        # Normalize so start ≤ end (− strand flips the values)
        ex_rel_start = min(ex_rel_a, ex_rel_b)
        ex_rel_end   = max(ex_rel_a, ex_rel_b)

        # ── Intron-pair boundary (gene-relative, 0-based inclusive) ──────────
        _, ib_g_start, ib_g_end = parse_coord(row["exon_boundary"])
        ib_rel_a = to_gene_rel(ib_g_start, gc)
        ib_rel_b = to_gene_rel(ib_g_end,   gc)
        intron_start = min(ib_rel_a, ib_rel_b)
        intron_end   = max(ib_rel_a, ib_rel_b)

        # ── 500 bp window centred on exon midpoint ────────────────────────────
        exon_mid  = (ex_rel_start + ex_rel_end) // 2
        win_start = max(0, exon_mid - half)
        win_end   = win_start + SJ_WIN
        if win_end > gc["length"]:       # clamp to gene boundary
            win_end   = gc["length"]
            win_start = max(0, win_end - SJ_WIN)

        # ── 25 kb primary window centred on exon midpoint ────────────────────
        # Shifted to fit inside the gene; padded by prep_01 when gene < 25 kb.
        p_half = P_WIN_LEN // 2
        wp_start = max(0, exon_mid - p_half)
        wp_end   = wp_start + P_WIN_LEN
        if wp_end > gc["length"]:
            wp_end   = gc["length"]
            wp_start = max(0, wp_end - P_WIN_LEN)
        wp_actual_len = wp_end - wp_start    # ≤ P_WIN_LEN; < P_WIN_LEN iff gene shorter

        # ── Sequence extraction ───────────────────────────────────────────────
        seq = fetch_window_seq(fa, gc, win_start, win_end)

        # Encode; zero-pad if the window was smaller than SJ_WIN at a boundary
        seq_enc = [ENCODING.get(b.upper(), 0) for b in seq]
        seq_enc += [0] * (SJ_WIN - len(seq_enc))

        # ── Annotation: 1 where the cassette exon is, 0 elsewhere ────────────
        annot = [0] * SJ_WIN
        local_start = max(0, ex_rel_start - win_start)
        local_end   = min(SJ_WIN, ex_rel_end - win_start + 1)   # exclusive
        for i in range(local_start, local_end):
            annot[i] = 1

        # ── Store ─────────────────────────────────────────────────────────────
        sj_dict[eid] = torch.tensor([seq_enc, annot], dtype=torch.long)

        # All coords below are PRIMARY-WINDOW-relative (0-based, 5′→3′).
        sj_rows.append({
            "event_id":    eid,
            "sj_st_indx":  win_start - wp_start,
            "sj_end_indx": win_end   - wp_start,
        })

        coord_rows.append({
            "event_id":     eid,
            "gene_id":      gid,
            "exon_start":   ex_rel_start - wp_start,
            "exon_end":     ex_rel_end   - wp_start,
            "intron_start": intron_start - wp_start,
            "intron_end":   intron_end   - wp_start,
            "gene_length":  gc["length"],
            # Filter in CellSpliceNet/src/data/dataloader.py keeps rows with
            # p_seq_len < max_prime_seq_len (default 25000). Cap at 24999 so
            # every windowed event passes; we never feed >P_WIN_LEN anyway.
            "p_seq_len":    min(wp_actual_len, P_WIN_LEN - 1),
            "sj_seq_len":   SJ_WIN,
            # Primary-window genomic offset (within gene), needed by prep_01
            # to fetch the matching 25 kb of sequence:
            "window_start_in_gene": wp_start,
            "window_end_in_gene":   wp_end,
        })

    fa.close()
    print(f"  Encoded {len(sj_dict)} events, skipped {skipped} (gene not in GTF)")

    # 4. Save outputs
    torch.save(sj_dict, OUT_DIR / "enc_sj_data.pth")
    pd.DataFrame(sj_rows).to_csv(OUT_DIR / "spliceregion_inds.csv", index=False)
    pd.DataFrame(coord_rows).to_csv(OUT_DIR / "events_coordinates.csv", index=False)
    print(f"Saved → {OUT_DIR}/enc_sj_data.pth")
    print(f"Saved → {OUT_DIR}/spliceregion_inds.csv")
    print(f"Saved → {OUT_DIR}/events_coordinates.csv")


if __name__ == "__main__":
    main()
