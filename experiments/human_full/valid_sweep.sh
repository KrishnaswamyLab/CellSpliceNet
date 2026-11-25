#!/bin/bash
# Re-rank every saved checkpoint (step_*, best, final) of the runs matching PATTERN on a fixed seeded subsample
# of the validation rows (gtex_valid_eval.py), e.g. after the capped-validation checkpoint selection.
#   PATTERN=hg_gene_deltacellsplicenet sbatch experiments/human_full/valid_sweep.sh
#SBATCH --job-name=csn_valid_sweep
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
source experiments/common/env.sh
set -eo pipefail
export RESULTS_CSV="${RESULTS_CSV:-$(python -c "from experiments.common.paths import out_dir; print(out_dir('human_full') / 'gtex_valid_reselect.csv')")}"
rm -f "$RESULTS_CSV"
for R in $(ls -d "${CSN_RUNS:-outputs}"/*/*01Feb2025_${PATTERN:?}*); do
  for ck in "$R"/model/step_*_model.pth "$R"/model/best_validation_model.pth "$R"/model/final_model.pth; do
    [ -f "$ck" ] || continue
    RUN_LABEL=$(basename "$R") CKPT="$ck" python experiments/human_full/gtex_valid_eval.py
  done
done
