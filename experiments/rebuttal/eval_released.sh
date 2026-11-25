#!/bin/bash
# Which split did the released C. elegans checkpoint train on, and did it see the control exons?
# Scores the released checkpoint (<data_root>/c_elegans/checkpoints/cellsplicenet_released.pth, or CKPT) on the
# valid and test sets of the row-level and grouped splits, and on the training rows of the row split with
# controls ('rows_train', the 'alltypes_trainastest' check). The highest valid rho marks its training split.
#SBATCH --job-name=csn_released
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
source experiments/common/env.sh
set -eo pipefail
CKPT="${CKPT:-$(python -c 'from experiments.common.paths import released_checkpoint; print(released_checkpoint())')}"
OUT=$(python -c "from experiments.common.paths import out_dir; print(out_dir('rebuttal', 'released'))")
CFG() { python -c "from experiments.common.paths import split_config; print(split_config('$1', 'c_elegans'))"; }
for sp in ${SPLITS:-rows_se event_sample1 gene_sample1 cell_sample1 rows}; do
  for part in valid test; do
    python src/eval.py --config_fname "$(CFG $sp)" --ckpts "$CKPT" --split $part --batch_size 8 --sfgenes 243 \
        --structure_length 500 --out_dir "$OUT/$sp"
  done
done
# training rows of the row split (with controls) scored as a test set
TRAIN_AS_TEST=$(python -c "
from experiments.common.paths import derive_config, out_dir, split_config, split_csv
print(derive_config(split_config('rows', 'c_elegans'), out_dir('rebuttal', 'released', 'rows_train') / 'data_config.ini',
                    test_data_file=split_csv('rows', 'train', 'c_elegans')))")
python src/eval.py --config_fname "$TRAIN_AS_TEST" --ckpts "$CKPT" --split test --batch_size 8 --sfgenes 243 \
    --structure_length 500 --out_dir "$OUT/rows_train"
