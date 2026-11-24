# src-ablation (archived, not imported)

Arman Afrasiyabi's 2025 ablation code, kept for reference. The maintained ablation models are in
`src/model/seqonly_baseline.py`.

`data/prep/prep_02..06_*` built the held-out event / gene / neuron worm splits. These scripts use one
randomly sampled replicate per (event, neuron); the paper's per-neuron data average the replicates.
