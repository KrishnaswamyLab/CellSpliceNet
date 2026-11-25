#!/bin/bash
# Per-row predictions with src/eval.py for one or more checkpoints.
#   CFG=<data_config.ini | split name> CKPTS="run1/model/a.pth run2/model/b.pth" sbatch experiments/common/eval.sbatch
#   CFG=gene_k8 PATTERN=hg_gene_k8_ CKPT_NAMES="final_model best_validation_model" MAX_EVENTS=1500 sbatch ...
# Env: ORGANISM [human], CFG, CKPTS (explicit paths) or PATTERN (run-dir glob under outputs/, see
#      experiments/common/paths.py run_dirs) + CKPT_NAMES [best_validation_model], SPLIT [test] | valid,
#      MAX_EVENTS [0 = all rows], AMP, BATCH_SIZE [16], SFGENES [493 human / 243 worm], EXTRA (more flags)
# Writes preds_<ckpt>.csv into each run dir (or OUT_DIR).
#SBATCH --job-name=csn_eval
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
source experiments/common/env.sh
set -eo pipefail
ORGANISM="${ORGANISM:-human}"
CFG="${CFG:?}"; [ -f "$CFG" ] || CFG="${CSN_DATA_ROOT:-$REPO/dataset}/$ORGANISM/splits/$CFG/data_config.ini"
if [ "$ORGANISM" = human ]; then SF=493; else SF=243; fi
if [ -z "${CKPTS:-}" ]; then
  CKPTS=""
  for R in $(ls -d "${CSN_RUNS:-outputs}"/*/*01Feb2025_${PATTERN:?}* 2>/dev/null); do
    for N in ${CKPT_NAMES:-best_validation_model}; do [ -f "$R/model/$N.pth" ] && CKPTS="$CKPTS $R/model/$N.pth"; done
  done
fi
[ -n "$CKPTS" ] || { echo "no checkpoints" >&2; exit 1; }
export SEQONLY_SYNTH_SHAPES="${SEQONLY_SYNTH_SHAPES:-1}"
python src/eval.py --config_fname "$CFG" --ckpts $CKPTS --split "${SPLIT:-test}" --max_events "${MAX_EVENTS:-0}" \
    --amp "${AMP:-}" --batch_size "${BATCH_SIZE:-16}" --sfgenes "${SFGENES:-$SF}" ${OUT_DIR:+--out_dir "$OUT_DIR"} ${EXTRA:-}
