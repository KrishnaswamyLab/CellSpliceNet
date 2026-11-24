# Archived: Arman Afrasiyabi's ablation code (not imported anywhere)

Kept for provenance of the published modality ablations and of the grouped (event / gene / neuron)
worm splits. It is a full copy of the 2025 `src/` with ablation switches
(`model/ztransformer_abilation.py`, `main_abilation.py`, `utils/validation_ablation.py`).

- Current ablation models live in `src/model/seqonly_baseline.py`. They zero modalities at token
  level, the same convention as `ztransformer_abilation.py`, but as wrapper modules on the
  maintained model.
- `data/prep/prep_02..06_*` are the scripts that produced the `12Jan2026_{event,gene,neuron}based_single`
  worm splits. They take **one randomly sampled replicate** per (event, neuron)
  (`groupby('neuron').sample(n=1)`), whereas the paper's row-level data average the replicates.
  The refactored split builder replaces them (docs/REFACTOR_PLAN.md, section 3).
