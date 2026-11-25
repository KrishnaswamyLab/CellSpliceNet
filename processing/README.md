# Data processing

All scripts read from and write to the data root: `--data_root`, else `$CSN_DATA_ROOT`, else
`<repo>/dataset`. Run them from the repo root. The data root contains `c_elegans/` and `human/`, each
with `raw/`, `processed/` and `events.csv`.

**Human (GTEx / ASCOT)**
1. `python processing/human/coords_and_junctions.py` writes the 500 bp splice-junction windows,
   `spliceregion_inds.csv` and `events_coordinates.csv`.
2. `python processing/human/sequences.py` writes the 25 kb sequence windows (`encoded_sequences.pth`).
3. `python processing/human/build_events.py` writes `human/events.csv`.
4. Structure: see below, with `--organism human`.
5. Expression: `sbatch processing/human/expression/run_all_tissues.sbatch` writes one scatter file per tissue.

**C. elegans**
1. `python processing/c_elegans/build_events.py` writes `c_elegans/events.csv`.
2. Expression: `processing/c_elegans/expression/pp01`–`pp04`, in order (`run.job`). MAGIC imputation,
   then per-neuron moments, then mutual-information graphs, then scattering.
3. Structure: see below, with `--organism c_elegans --max-sequence-len 500`.

**Structure (both organisms)**
1. `processing/structure/fold.py` folds with RNAfold. Human uses the 500 bp splice-junction window;
   C. elegans uses the intron-exon-intron region of the gene, capped at 500 bp around the exon.
2. `processing/structure/scatter_structures.py` computes the graph scattering (GPU).
3. `processing/structure/collect.py` writes `processed/structure/structure_scattering_dict.pkl`.

Splits are made from `events.csv` by `splits/make_splits.py`.
