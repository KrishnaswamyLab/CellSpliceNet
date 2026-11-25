#!/bin/bash
# Run one experiment script on a CPU node. Resources are set on the command line, e.g.
#   sbatch -c 8 --mem 200G -t 6:00:00 experiments/common/py.sbatch experiments/human_gen/track1_transfer.py
#   (environment variables the script reads are passed through)
#SBATCH --job-name=csn_py
#SBATCH -p day
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
source experiments/common/env.sh
set -eo pipefail
echo "[py] $*"
python -u "$@"
