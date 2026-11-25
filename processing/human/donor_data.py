#!/usr/bin/env python3
"""Per-sample (donor-level) GTEx data: cassette-exon PSI from split reads plus gene expression.

Exons: SD of ASCOT PSI over the training tissues >= 0.05 and measured in >= 5 tissues
(processed/event_variability.csv; selection never looks at held-out tissues), ~10.7k exons.
For every sample of the 49 tissues:
  psi   PSI from split reads, inc = mean(inc1, inc2), exc = skipping junction (see junctions.py);
        NaN when inc + exc < 1
  tot   inc + exc
  expr  log1p(CPM) of the genes whose tissue pseudobulk is >= 1 in some tissue, plus all 493 model
        splicing factors

Inputs:  processed/event_variability.csv (event_variability.py),
         processed/expression/tissue_pseudobulk.csv.gz (expression/pseudobulk.py),
         processed/expression/scatter_493/exact_order_of_splice_factors.csv,
         raw/gtex_counts/exported_counts/<tissue>/{k_j_counts.tsv.gz, geneCounts.tsv.gz, sampleAnnotation.tsv}
Output:  <data_root>/human/processed/donor/
         meta_events.csv, meta_genes.csv (gene, is_sf), meta_samples.csv (sample, donor, sex, tissue, folder, row)
         <tissue folder>.npz: psi float16 [samples x events], tot uint16, expr float16 [samples x genes]
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from processing.common import organism_paths  # noqa: E402
from processing.human.expression.tissue_folder_map import FOLDER_TO_ASCOT  # noqa: E402
from processing.human.junctions import counts_dir, junction_keys, read_junctions, tissue_folders  # noqa: E402

RAW, PROCESSED, OUT = organism_paths("human")
DST = OUT / "donor"
EC = counts_dir(RAW)


def find(rel):
    """An earlier pipeline output: from --out_dir if it was written there (test runs), else processed/."""
    return OUT / rel if (OUT / rel).exists() else PROCESSED / rel


def setup():
    ev = pd.read_csv(find("event_variability.csv"), index_col=0)
    events = ev.index[(ev.sd_train_tissues >= 0.05) & (ev.n_tissues >= 5)]
    all_events = pd.read_csv(PROCESSED.parent / "events.csv", usecols=["event_id"]).event_id.unique()
    asc, J = junction_keys(RAW, all_events)
    events = [e for e in events if e in asc.index]
    JJ = {k: v.loc[events].values for k, v in J.items()}
    sf = [l.strip() for l in open(PROCESSED / "expression/scatter_493/exact_order_of_splice_factors.csv") if l.strip()]
    pb = pd.read_csv(find("expression/tissue_pseudobulk.csv.gz"), index_col=0)
    pb.index = pb.index.str.split(".").str[0]
    pb = pb[~pb.index.duplicated()]
    genes = sorted(set(pb.index[pb.max(1) >= 1.0]) | set(sf))
    return events, JJ, sf, genes


EVENTS, JJ, SF, GENES = setup()
NEED = set(np.concatenate(list(JJ.values())))


def one(folder):
    tissue = FOLDER_TO_ASCOT[folder.split("--")[0]]
    K = read_junctions(EC / folder / "k_j_counts.tsv.gz", NEED)
    G = pd.read_csv(EC / folder / "geneCounts.tsv.gz", sep="\t", index_col=0)
    G.index = G.index.str.split(".").str[0]
    G = G[~G.index.duplicated()]
    a = pd.read_csv(EC / folder / "sampleAnnotation.tsv", sep="\t").set_index("RNA_ID")
    S = [s for s in K.columns if s in G.columns and s in a.index]
    K = K[S]
    inc = (K.reindex(JJ["inc1"]).fillna(0).values + K.reindex(JJ["inc2"]).fillna(0).values) / 2
    exc = K.reindex(JJ["skip"]).fillna(0).values
    tot = inc + exc
    psi = np.where(tot >= 1, inc / np.maximum(tot, 1e-9), np.nan)
    L = np.log1p(G[S] / G[S].sum(0) * 1e6).reindex(GENES).fillna(0)
    np.savez(DST / (folder.split("--")[0] + ".npz"), psi=psi.T.astype(np.float16),
             tot=np.clip(np.round(tot.T), 0, 65535).astype(np.uint16), expr=L.T.values.astype(np.float16))
    print(f"  {tissue}: samples {len(S)}  median reads/exon {np.median(tot):.0f}", flush=True)
    return pd.DataFrame({"sample": S, "donor": a.loc[S, "INDIVIDUAL_ID"].values, "sex": a.loc[S, "SEX"].values,
                         "tissue": tissue, "folder": folder.split("--")[0], "row": np.arange(len(S))})


def main():
    DST.mkdir(parents=True, exist_ok=True)
    print(f"events {len(EVENTS):,}  junctions {len(NEED):,}  genes {len(GENES):,} "
          f"(splicing factors among them: {len(set(SF) & set(GENES))})", flush=True)
    pd.DataFrame({"event_id": EVENTS}).to_csv(DST / "meta_events.csv", index=False)
    pd.DataFrame({"gene": GENES, "is_sf": [g in set(SF) for g in GENES]}).to_csv(DST / "meta_genes.csv", index=False)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as p:
        M = pd.concat(p.map(one, tissue_folders(RAW)), ignore_index=True)
    M.to_csv(DST / "meta_samples.csv", index=False)
    print(f"samples {len(M):,} donors {M.donor.nunique():,} -> {DST}", flush=True)


if __name__ == "__main__":
    main()
