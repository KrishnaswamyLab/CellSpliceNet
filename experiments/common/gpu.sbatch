#!/bin/bash
# Run one experiment script on a GPU node, e.g.
#   PATTERN=hg_gene_x_cell_em_csn_noexpr sbatch experiments/common/gpu.sbatch experiments/human_gen/extract_embeddings.py
#SBATCH --job-name=csn_gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
source experiments/common/env.sh
set -eo pipefail
export SEQONLY_SYNTH_SHAPES="${SEQONLY_SYNTH_SHAPES:-1}"
echo "[gpu] $*"
python -u "$@"
