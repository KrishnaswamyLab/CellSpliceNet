#!/bin/bash
# Portable sbatch wrapper for the step-based trainer (src/train_full.py).
#
# Edit the SLURM directives for your cluster/account, then: sbatch src/train_full.sh
# All data paths come from data_config.ini ([files]).
#
#SBATCH --job-name=cellsplice_full
#SBATCH -p pi_krishnaswamy
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=cellsplice_full_%j.out

source ~/.bashrc
module load miniconda
conda activate mioflow
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CONFIG_FNAME="${CONFIG_FNAME:-$REPO_DIR/dataset/human/data_config.ini}"

DATA_TAG="${DATA_TAG:-gtex}"       # gtex/human -> human config; worm tags -> c_elegans
SFGENES="${SFGENES:-493}"          # 493 = GTEx, 243 = worm
BATCH_SIZE="${BATCH_SIZE:-8}"
N_STEPS="${N_STEPS:-300000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
TIME_BUDGET_S="${TIME_BUDGET_S:-172000}"

python src/train_full.py \
    --data_tag             "$DATA_TAG" \
    --config_fname         "$CONFIG_FNAME" \
    --sfgenes              "$SFGENES" \
    --batch_size           "$BATCH_SIZE" \
    --n_steps              "$N_STEPS" \
    --eval_every_iteration "$EVAL_EVERY" \
    --time_budget_s        "$TIME_BUDGET_S"
