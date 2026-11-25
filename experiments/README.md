# Experiments

Scripts behind the generalization experiments, the reviewer-response analyses and the method comparisons.
Run everything from the repo root. Data come from the data root (`$CSN_DATA_ROOT`, see `processing/README.md`);
results go to `$CSN_EXP_OUT` (default `outputs/experiments`), training runs to `outputs/<model>/` (`$CSN_RUNS`
to read runs stored elsewhere). Slurm scripts write to `logs/` and run `$CSN_ACTIVATE` first
(e.g. `export CSN_ACTIVATE="module load miniconda; conda activate csn"`).

| Folder | What |
|---|---|
| `common/` | `train.sbatch` (`src/train_full.py` on a data-root split), `eval.sbatch` (`src/eval.py`), `py.sbatch` / `gpu.sbatch` (run one script), `paths.py` |
| `human_gen/` | Human (GTEx / ASCOT) generalization to new exons and tissues: feature / tissue baselines, noise ceiling, base + tissue-delta probes and heads, donor-level expression models (`track1_transfer.py`), reports |
| `human_full/` | Full GTEx training checks: seeded-subsample evaluation, validation re-selection, cross-organism transfer, embeddings |
| `worm_ablations/` | C. elegans ablations: expression denoiser / graph / encoder, structure window, ViT, SeqOnly, grouped splits (`worm_train.py`, `launch.sh`, `eval.sbatch`, `dump_preds.py`, perturbation tests) |
| `rebuttal/` | Reviewer analyses: released-checkpoint checks, held-out split diagnostics, genuine-SE evaluation, descriptives and PHATE figures |
| `comparisons/alphagenome/` | Zero-shot AlphaGenome splicing predictions (needs `ALPHAGENOME_API_KEY`) vs CellSpliceNet |

Common commands:

```bash
# train / evaluate on a split (human by default; ORGANISM=c_elegans for worm)
SPLIT=gene_k8 MODEL=deltacellsplicenet SEED=1 PSI_HEAD=twohead AMP=bf16 sbatch experiments/common/train.sbatch
CFG=gene_k8 PATTERN=hg_gene_k8_deltacellsplicenet MAX_EVENTS=1500 sbatch experiments/common/eval.sbatch

# human generalization
bash experiments/human_gen/launch_pilot2.sh                       # pilot runs + evals
sbatch experiments/common/py.sbatch experiments/human_gen/split_baselines.py gene_k8
sbatch experiments/common/py.sbatch experiments/human_gen/pilot_report.py
bash experiments/human_gen/launch_stage_a.sh                      # base + tissue-delta arms

# C. elegans ablations
bash experiments/worm_ablations/launch.sh multiseed
RUNS="outputs/experiments/worm_ablations/runs/expression/*" sbatch experiments/worm_ablations/eval.sbatch
python experiments/worm_ablations/aggregate_multiseed_eval.py
```

Split names follow `splits/` (`gene`, `chrom`, `cell`, `gene_x_cell`, `*_k8`, `rows`, `rows_se`, `*_sample1` ...);
the older names (`orig8`, `tissue`, `combo`, `SE`, `eventbased` ...) are accepted by `experiments/common/paths.py`,
so runs trained before these scripts can still be found and scored.
