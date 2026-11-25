#!/bin/bash
# Reviewer "genuine SE" evaluation inputs for one seed (sbatch --array=1-3):
#   1. train the SE-only anchor (scatter / magic / mi_knn3 / 500 nt; 15k steps, eval every 1k) on rows_se
#   2. predict its rows_se test set, and the all-types anchor's (<runs>/expression/magic_mi_knn3_scatter__seed<S>,
#      launch.sh multiseed) test set of the split with controls
#   ARMAN=1 instead predicts Arman's event-held-out checkpoint on event_sample1 (for eval_arman_eventbased.py).
# Then: python experiments/rebuttal/genuine_se/psi_range_and_residual_eval.py --seed <S>
#SBATCH --job-name=csn_genuine_se
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
source experiments/common/env.sh
set -eo pipefail
OUT=$(python -c "from experiments.common.paths import out_dir; print(out_dir('rebuttal', 'arman', 'event_sample1'))")
if [ "${ARMAN:-0}" = 1 ]; then
  CK=$(python -c "from experiments.common.paths import WORM; print(WORM / 'checkpoints/arman_12Jan2026/event_sample1_epoch270.pth')")
  CFG=$(python -c "from experiments.common.paths import split_config; print(split_config('event_sample1', 'c_elegans'))")
  python src/eval.py --config_fname "$CFG" --ckpts "$CK" --batch_size 8 --sfgenes 243 --out_dir "$OUT"
  exit 0
fi
SEED="${SEED:-$SLURM_ARRAY_TASK_ID}"
RUNS=$(python -c "from experiments.worm_ablations.common import abl_runs; print(abl_runs())")
R="$RUNS/se_only/magic_mi_knn3_scatter__seed$SEED"
SPLIT=rows_se OUTPUT_DIR="$R" DATASET_TAG=worm_ablation BATCH_SIZE=8 N_STEPS=15000 TIME_BUDGET_S=9000 EVAL_EVERY=1000 \
  SAVE_STEP_CKPT_EVERY=0 SEED=$SEED python experiments/worm_ablations/worm_train.py
python experiments/worm_ablations/dump_preds.py "$R"
SPLIT=rows python experiments/worm_ablations/dump_preds.py "$RUNS/expression/magic_mi_knn3_scatter__seed$SEED"
