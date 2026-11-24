#!/bin/bash
# GTEx expression scatter coefficients for all 49 tissues (one array task per tissue).
#   sbatch processing/human/expression/run_all_tissues.sbatch          (from the repo root)
# The gene order is fixed first (--init-only), then every tissue reuses it.
#SBATCH --job-name=csn_gtex_expr
#SBATCH --array=0-48
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -eo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
S=processing/human/expression/run_bulk_expression.py
TISSUES=($(python -c "from processing.human.expression.tissue_folder_map import FOLDER_TO_ASCOT as m; print(' '.join(sorted(m)))"))
OUT=$(python -c "from processing.common import default_data_root as d; print(d() / 'human/processed/expression/scatter_493')")
mkdir -p "$OUT"
{ flock 200; [ -s "$OUT/exact_order_of_splice_factors.csv" ] || python $S --allow-missing-gobp --init-only; } 200>"$OUT/.setup.lock"
python $S --allow-missing-gobp --from-existing-order --device cpu --tissue "${TISSUES[$SLURM_ARRAY_TASK_ID]}"
