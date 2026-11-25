# Sourced by the experiment sbatch / shell scripts. Submit them from the repo root.
#   CSN_ACTIVATE   command that activates the Python environment, e.g.
#                  export CSN_ACTIVATE="module load miniconda; conda activate csn"
#   CSN_DATA_ROOT  data root (default <repo>/dataset)
#   CSN_EXP_OUT    experiment outputs (default <repo>/outputs/experiments)
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f src/train_full.py ]; then echo "run from the CellSpliceNet repo root" >&2; exit 1; fi
REPO=$PWD
if [ -n "${CSN_ACTIVATE:-}" ]; then eval "$CSN_ACTIVATE"; fi
export PYTHONUNBUFFERED=1 PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
