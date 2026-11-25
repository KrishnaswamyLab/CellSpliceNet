#!/bin/bash
# Valid + test metrics (worm_eval.py) for ablation runs, one row per run in <out>/worm_ablations/eval_rows/.
#   RUNS="<runs>/expression/*__seed1 <runs>/structure/stc_750__seed1" sbatch experiments/worm_ablations/eval.sbatch
#   MANIFEST=list.txt sbatch --array=0-41%8 experiments/worm_ablations/eval.sbatch   (one run dir per line)
# Then: python experiments/worm_ablations/aggregate_multiseed_eval.py
# Env passed through: SPLIT (runs made before the repo: the split they were trained on), CKPT_NAME, DUMP_PREDS=1
#SBATCH --job-name=csn_ablation_eval
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
source experiments/common/env.sh
set -eo pipefail
ROWS=$(python -c "from experiments.common.paths import out_dir; print(out_dir('worm_ablations', 'eval_rows'))")
if [ -n "${MANIFEST:-}" ]; then RUNS=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$MANIFEST"); fi
for R in ${RUNS:?set RUNS or MANIFEST}; do
  [ -f "$R/model/${CKPT_NAME:-best_validation_model.pth}" ] || { echo "[skip] no checkpoint in $R"; continue; }
  out="$ROWS/$(basename "$(dirname "$R")")__$(basename "$R").csv"; rm -f "$out"
  RUN_DIR="$R" RESULTS_CSV="$out" DEVICE=cuda python experiments/worm_ablations/worm_eval.py
done
