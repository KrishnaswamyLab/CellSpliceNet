#!/bin/bash
# One C. elegans ablation run with worm_train.py (all knobs are env vars, see worm_train.py), e.g.
#   OUTPUT_DIR=<runs>/expression/magic_mi_knn3_scatter__seed1 SEED=1 sbatch experiments/worm_ablations/train.sbatch
#   MODEL=csn_seqonly SPLIT=event_sample1 SEQONLY_SYNTH_SHAPES=1 SEED=1 OUTPUT_DIR=... sbatch ...
# Defaults (rebuttal protocol): 32k steps, batch 8, Adam 2e-5, full validation pass every 2000 steps,
# best checkpoint by validation loss. launch.sh submits the ablation sets.
#SBATCH --job-name=csn_ablation
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
source experiments/common/env.sh
set -eo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export N_STEPS="${N_STEPS:-32000}" EVAL_EVERY="${EVAL_EVERY:-2000}" VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-100000}"
export TIME_BUDGET_S="${TIME_BUDGET_S:-36000}" SAVE_STEP_CKPT_EVERY="${SAVE_STEP_CKPT_EVERY:-0}" BATCH_SIZE="${BATCH_SIZE:-8}"
export SEQONLY_SYNTH_SHAPES="${SEQONLY_SYNTH_SHAPES:-0}"
echo "[ablation] MODEL=${MODEL:-deltacellsplicenet} SPLIT=${SPLIT:-rows} SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-}"
python experiments/worm_ablations/worm_train.py
