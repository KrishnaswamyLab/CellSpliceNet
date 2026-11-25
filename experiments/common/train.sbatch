#!/bin/bash
# Train on a split of the data root with src/train_full.py.
#   SPLIT=gene_k8 MODEL=deltacellsplicenet SEED=1 sbatch experiments/common/train.sbatch
# Env (defaults in brackets):
#   ORGANISM [human] | c_elegans          SPLIT  folder under <data_root>/<organism>/splits (or a data_config.ini path)
#   MODEL, SEED, PSI_HEAD [log] | linear | twohead, AMP [''] | bf16, TAG [''] (appended to the run name)
#   N_STEPS [80000], BATCH_SIZE [8], EVAL_EVERY [2000], VALID_MAX_BATCHES [200], SAVE_STEP_CKPT_EVERY [10],
#   TIME_BUDGET_S [82800], SFGENES [493 human / 243 worm], EXTRA (more train_full.py flags)
# Run dir: outputs/<MODEL>/<time key>01Feb2025_<hg|worm>_<SPLIT>_<MODEL><TAG>_s<SEED>/
#SBATCH --job-name=csn_train
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 6
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
source experiments/common/env.sh
set -eo pipefail
ORGANISM="${ORGANISM:-human}"; SPLIT="${SPLIT:?}"; MODEL="${MODEL:?}"; SEED="${SEED:?}"
if [ -f "$SPLIT" ]; then CFG="$SPLIT"; NAME=$(basename "$(dirname "$SPLIT")")
else CFG="${CSN_DATA_ROOT:-$REPO/dataset}/$ORGANISM/splits/$SPLIT/data_config.ini"; NAME=$SPLIT; fi
[ -f "$CFG" ] || { echo "no config $CFG" >&2; exit 1; }
if [ "$ORGANISM" = human ]; then P=hg; SF=493; else P=worm; SF=243; fi
export MODEL SEQONLY_SYNTH_SHAPES="${SEQONLY_SYNTH_SHAPES:-1}" AMP="${AMP:-}" PSI_HEAD="${PSI_HEAD:-log}"
echo "[train] CFG=$CFG MODEL=$MODEL SEED=$SEED PSI_HEAD=$PSI_HEAD AMP=$AMP N_STEPS=${N_STEPS:-80000}"
python src/train_full.py \
    --data_tag             "${P}_${NAME}_${MODEL}${TAG:-}_s${SEED}" \
    --model                "$MODEL" \
    --config_fname         "$CFG" \
    --sfgenes              "${SFGENES:-$SF}" \
    --seed                 "$SEED" \
    --batch_size           "${BATCH_SIZE:-8}" \
    --n_steps              "${N_STEPS:-80000}" \
    --eval_every_iteration "${EVAL_EVERY:-2000}" \
    --valid_max_batches    "${VALID_MAX_BATCHES:-200}" \
    --save_step_ckpt_every "${SAVE_STEP_CKPT_EVERY:-10}" \
    --time_budget_s        "${TIME_BUDGET_S:-82800}" \
    ${EXTRA:-}
