#!/usr/bin/env python3
"""Per-tissue split-read counts of every cassette exon, from the GTEx per-sample junction counts.

For an ASCOT exon (exon_location s-e, exon_boundary b1-b2):
    skipping junction   (b1, b2)            (exon_boundary is exactly the skipping intron)
    inclusion junctions (b1, s-1), (e+1, b2)
    inc = mean(inc1, inc2), exc = skip, PSI = inc / (inc + exc)
The samples of each tissue are split at random into halves A and B (seeded by the tissue name), for
split-half reliability; F = all samples.

Inputs:  <data_root>/human/events.csv, raw/ascot/gtex_psi.csv.gz,
         raw/gtex_counts/exported_counts/<tissue>--GRCh38--gencode29/k_j_counts.tsv.gz (FRASER export)
Output:  <data_root>/human/processed/junction_counts.csv.gz
         event_id, tissue, n_samples, inc_A, exc_A, inc_B, exc_B, inc_F, exc_F
"""
import os
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from processing.common import organism_paths  # noqa: E402
from processing.human.expression.tissue_folder_map import FOLDER_TO_ASCOT  # noqa: E402


def counts_dir(raw):
    return Path(raw) / "gtex_counts/exported_counts"


def tissue_folders(raw):
    return sorted(d for d in os.listdir(counts_dir(raw)) if d.split("--")[0] in FOLDER_TO_ASCOT)


def junction_keys(raw, events):
    """ASCOT coordinates of `events` and their junction keys 'chrom:start:end' (skip, inc1, inc2)."""
    asc = pd.read_csv(Path(raw) / "ascot/gtex_psi.csv.gz", encoding="utf-8-sig",
                      usecols=["exon_id", "exon_location", "exon_boundary"]).set_index("exon_id").loc[events]
    loc = asc.exon_location.str.extract(r"(chr[^:]+):(\d+)-(\d+)")
    bnd = asc.exon_boundary.str.extract(r"(chr[^:]+):(\d+)-(\d+)")
    s, e, b1, b2 = (x.astype(int) for x in (loc[1], loc[2], bnd[1], bnd[2]))
    ch = loc[0]
    J = {"skip": ch + ":" + b1.astype(str) + ":" + b2.astype(str),
         "inc1": ch + ":" + b1.astype(str) + ":" + (s - 1).astype(str),
         "inc2": ch + ":" + (e + 1).astype(str) + ":" + b2.astype(str)}
    return asc, J


def read_junctions(path, need):
    """Rows of a k_j_counts file whose junction key is in `need`; index = key, columns = samples."""
    parts = []
    for ch in pd.read_csv(path, sep="\t", chunksize=100_000):
        key = ch.seqnames.astype(str) + ":" + ch.start.astype(str) + ":" + ch.end.astype(str)
        m = key.isin(need)
        if m.any():
            x = ch.loc[m].iloc[:, 4:]
            x.index = key[m]
            parts.append(x)
    return pd.concat(parts).groupby(level=0).sum()      # strand duplicates, if any


def _one(job):
    raw, folder, events, J, need = job
    tissue = FOLDER_TO_ASCOT[folder.split("--")[0]]
    X = read_junctions(counts_dir(raw) / folder / "k_j_counts.tsv.gz", need)
    n = X.shape[1]
    perm = np.random.default_rng(zlib.crc32(tissue.encode())).permutation(n)
    A, B = X.columns[perm[: n // 2]], X.columns[perm[n // 2:]]
    sums = {"A": X[A].sum(1), "B": X[B].sum(1), "F": X.sum(1)}
    out = pd.DataFrame(index=events)
    out["tissue"] = tissue
    out["n_samples"] = n
    for h, v in sums.items():
        out[f"inc_{h}"] = (J["inc1"].map(v).fillna(0).values + J["inc2"].map(v).fillna(0).values) / 2
        out[f"exc_{h}"] = J["skip"].map(v).fillna(0).values
    print(f"  {tissue}: samples {n}, junction rows {len(X):,}", flush=True)
    return out.rename_axis("event_id").reset_index()


def main():
    raw, processed, out = organism_paths("human")
    events = pd.read_csv(processed.parent / "events.csv", usecols=["event_id"]).event_id.unique()
    asc, J = junction_keys(raw, events)
    need = set(pd.concat(J.values()))
    print(f"events {len(asc):,}  junctions {len(need):,}", flush=True)
    jobs = [(raw, f, asc.index, J, need) for f in tissue_folders(raw)]
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as p:
        D = pd.concat(p.map(_one, jobs), ignore_index=True)
    D.to_csv(out / "junction_counts.csv.gz", index=False)
    print(f"junction_counts.csv.gz: {len(D):,} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
