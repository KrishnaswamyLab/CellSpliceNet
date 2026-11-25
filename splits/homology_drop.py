#!/usr/bin/env python3
"""Rebuild splits/frozen/human_chrom_homology_drop.csv.gz (the homology filter of the human chrom split).

Exons (+-100 nt, transcript strand) are assigned to train / valid / test by chromosome (specs/human/chrom.json).
A train exon with a megablast hit (e < 1e-10, alignment >= 50 nt) to a test or valid exon is dropped from
train; a valid exon with such a hit to a test exon is dropped from valid.

    python splits/homology_drop.py [--out splits/frozen/human_chrom_homology_drop.csv.gz] [--work DIR]
Needs BLAST+ (module load BLAST+/2.15.0-gompi-2022b, or BLAST_MODULE='' if blastn is on PATH).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from processing.common import default_data_root  # noqa: E402
from processing.human.genome import Genome, parse_interval  # noqa: E402

COMP = str.maketrans("ACGTN", "TGCAN")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--out", default=str(HERE / "frozen/human_chrom_homology_drop.csv.gz"))
    ap.add_argument("--work", default=None, help="folder for the FASTA / BLAST files (default: a temp dir)")
    a = ap.parse_args()
    root = Path(a.data_root) / "human"
    spec = json.load(open(HERE / "specs/human/chrom.json"))
    ev = pd.read_csv(root / "events.csv", usecols=["event_id", "chrom"]).drop_duplicates("event_id").set_index("event_id").chrom
    part = ev.map(lambda c: "test" if c in spec["test_chroms"] else ("valid" if c in spec["valid_chroms"] else "train"))
    asc = pd.read_csv(root / "raw/ascot/gtex_psi.csv.gz", usecols=["exon_id", "exon_strand", "exon_location"],
                      encoding="utf-8-sig").set_index("exon_id")
    g = Genome(root / "raw/hg38/GRCh38.p14.genome.fa")
    work = Path(a.work or tempfile.mkdtemp(prefix="homology_"))
    work.mkdir(parents=True, exist_ok=True)
    for k in ("train", "valid", "test"):
        with open(work / f"{k}.fa", "w") as f:
            for e in part.index[part == k]:
                ch, s, t = parse_interval(asc.exon_location[e])
                x = g.fetch(ch, s - 101, t + 100)
                f.write(f">{e}\n{x.translate(COMP)[::-1] if asc.exon_strand[e] == '-' else x}\n")
    nt = os.environ.get("SLURM_CPUS_PER_TASK", "4")
    mod = os.environ.get("BLAST_MODULE", "BLAST+/2.15.0-gompi-2022b")
    blast = "blastn -task megablast -evalue 1e-10 -outfmt '6 qseqid sseqid pident length bitscore' -max_target_seqs 5 -num_threads " + nt
    sh = ((f"module load {mod} && " if mod else "") + f"cd {work} && "
          "makeblastdb -in test.fa -dbtype nucl -out testdb >/dev/null && makeblastdb -in valid.fa -dbtype nucl -out validdb >/dev/null && "
          f"{blast} -query train.fa -db testdb > train_vs_test.tsv && {blast} -query valid.fa -db testdb > valid_vs_test.tsv && "
          f"{blast} -query train.fa -db validdb > train_vs_valid.tsv")
    subprocess.run(["bash", "-lc", sh], check=True)

    def hits(f):
        b = pd.read_csv(work / f, sep="\t", names=["q", "s", "pid", "alen", "bits"])
        return set(b.q[b.alen >= 50])

    drop_tr = hits("train_vs_test.tsv") | hits("train_vs_valid.tsv")
    drop_va = hits("valid_vs_test.tsv")
    out = pd.concat([pd.DataFrame({"event_id": sorted(drop_tr), "dropped_from": "train"}),
                     pd.DataFrame({"event_id": sorted(drop_va), "dropped_from": "valid"})], ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"homology filter: drop {len(drop_tr)} train, {len(drop_va)} valid exons -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
