#!/bin/bash
# Portable sbatch wrapper for the step-based trainer (src/train_full.py).
#
# Edit the SLURM directives for your cluster/account, set DATA_DIR (and
# SCATTER_DIR if your scatter coeffs live elsewhere), then: sbatch src/train_full.sh
# All data paths are passed as CLI flags, so nothing is hardcoded to one machine.
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

# Resolve the repo root from this script's location.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# --- Point these at your data (override on the command line / sbatch --export) ---
# Defaults assume data staged under <repo>/dataset/ (see args.py). Adjust freely.
DATA_DIR="${DATA_DIR:-$REPO_DIR/dataset/Alternative-Splicing/main}"
CONFIG_FNAME="${CONFIG_FNAME:-$DATA_DIR/data_config.ini}"
EXPRESSION_DATA_ROOT="${EXPRESSION_DATA_ROOT:-$DATA_DIR/train_data.csv}"
STRUCTURE_DATA_ROOT="${STRUCTURE_DATA_ROOT:-$DATA_DIR/structure_scattering_dict.pkl}"
SCATTER_DIR="${SCATTER_DIR:-$DATA_DIR/scatter_coeffs}"

# --- Run-budget knobs (CLI flags; see train_full.py for the full list) ---
SFGENES="${SFGENES:-493}"          # 493 = GTEx, 243 = worm
BATCH_SIZE="${BATCH_SIZE:-8}"
N_STEPS="${N_STEPS:-300000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
TIME_BUDGET_S="${TIME_BUDGET_S:-172000}"   # stop cleanly before the 48h walltime

python src/train_full.py \
    --config_fname         "$CONFIG_FNAME" \
    --expression_data_root "$EXPRESSION_DATA_ROOT" \
    --structure_data_root  "$STRUCTURE_DATA_ROOT" \
    --scatter_coeffs_dir   "$SCATTER_DIR" \
    --sfgenes              "$SFGENES" \
    --batch_size           "$BATCH_SIZE" \
    --n_steps              "$N_STEPS" \
    --eval_every_iteration "$EVAL_EVERY" \
    --time_budget_s        "$TIME_BUDGET_S"
