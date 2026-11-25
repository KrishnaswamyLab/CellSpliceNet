#!/bin/bash
# Submit sets of C. elegans ablation runs (train.sbatch). Run from the repo root.
#   bash experiments/worm_ablations/launch.sh <set>     DRY_RUN=1 prints the sbatch lines; SEEDS overrides seeds
# Sets:
#   structure        500 / 750 / 1000 nt structure windows at the expression anchor (magic, mi_knn3, scatter)
#   stc300           300 nt window                                                 [SEEDS 1-5]
#   expression       denoiser (magic / raw_log1p / knn_smooth), graph metric and encoder axes, one seed
#   multiseed        the expression matrix + structure 750/1000 over SEEDS            [42 1 2]
#   graphconstruct   binary-density / continuous-weighted scatter variants (mi_knn3, pearson, spearman)  [1 2 3]
#   vit              ViT baselines vitcsn_seq / vitcsn_all                            [1 2 3]
#   seqonly          SeqOnly on SPLITS (old or new names; default the three grouped splits)  [1]
#   fullmodel_grouped full model on SPLITS, matched to seqonly                       [1]
# Runs go to <runs>/<group>/<name>__seed<N> (<runs> = $CSN_ABL_RUNS or outputs/experiments/worm_ablations/runs).
set -eo pipefail
[ -f src/train_full.py ] || { echo "run from the repo root" >&2; exit 1; }
RUNS=$(python -c "from experiments.worm_ablations.common import abl_runs; print(abl_runs())")
S=experiments/worm_ablations/train.sbatch
submit() {  # job name, output dir, env assignments
  local name=$1 out=$2; shift 2
  local env="ALL,OUTPUT_DIR=$out,DATASET_TAG=$name"; for kv in "$@"; do env="$env,$kv"; done
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "sbatch -J csn_$name --export=$env $S"; else sbatch -J "csn_$name" --export="$env" "$S"; fi
}
abl() { python -c "from experiments.worm_ablations.common import EXPR_ABL; print(EXPR_ABL / '$1')"; }
expr_axes() {  # seed suffix ('' or __seedN), extra env
  local sfx=$1; shift
  for den in magic raw_log1p knn_smooth; do
    submit "exp_${den}_mi_knn3_scatter$sfx" "$RUNS/expression/${den}_mi_knn3_scatter$sfx" EXPRESSION_ENCODER=scatter EXPRESSION_ABLATION_ROOT=$(abl $den) EXPRESSION_METRIC=mi_knn3 "$@"
  done
  for m in mi_knn5 mi_knn10 mi_binned10 pearson spearman; do
    submit "exp_magic_${m}_scatter$sfx" "$RUNS/expression/magic_${m}_scatter$sfx" EXPRESSION_ENCODER=scatter EXPRESSION_ABLATION_ROOT=$(abl magic) EXPRESSION_METRIC=$m "$@"
  done
  for enc in $ENCODERS; do
    submit "exp_magic_mi_knn3_${enc}$sfx" "$RUNS/expression/magic_mi_knn3_${enc}$sfx" EXPRESSION_ENCODER=$enc EXPRESSION_ABLATION_ROOT=$(abl magic) EXPRESSION_METRIC=mi_knn3 "$@"
  done
}
case "${1:-}" in
  structure)
    for L in 500 750 1000; do submit "stc_$L" "$RUNS/structure/stc_$L" EXPRESSION_ABLATION_ROOT=$(abl magic) STRUCTURE_LEN=$L; done ;;
  stc300)
    for s in ${SEEDS:-1 2 3 4 5}; do submit "stc_300_seed$s" "$RUNS/structure/stc_300__seed$s" EXPRESSION_ABLATION_ROOT=$(abl magic) STRUCTURE_LEN=300 SEED=$s; done ;;
  expression)
    ENCODERS="mlp gnn" expr_axes "" ;;
  multiseed)
    for s in ${SEEDS:-42 1 2}; do
      ENCODERS="gnn mlp_node mlp_scalar" expr_axes "__seed$s" SEED=$s
      for L in 750 1000; do submit "stc_${L}_seed$s" "$RUNS/structure/stc_${L}__seed$s" EXPRESSION_ABLATION_ROOT=$(abl magic) STRUCTURE_LEN=$L SEED=$s; done
    done ;;
  graphconstruct)
    for v in "mi_knn3 binary_d0.3" "mi_knn3 binary_d0.5" "mi_knn3 binary_d0.7" "mi_knn3 continuous_weighted" \
             "pearson continuous_weighted" "pearson binary_d0.3" "spearman continuous_weighted" "spearman binary_d0.3"; do
      set -- $v
      for s in ${SEEDS:-1 2 3}; do
        submit "exp_magic_$1_$2__seed$s" "$RUNS/expression/magic_$1_$2__seed$s" EXPRESSION_ABLATION_ROOT=$(abl magic) EXPRESSION_METRIC=$1 SCATTER_SUBDIR=scatter_coeffs_$2 SEED=$s
      done
    done ;;
  vit)
    for m in vitcsn_seq vitcsn_all; do for s in ${SEEDS:-1 2 3}; do
      submit "vit_${m}_seed$s" "$RUNS/vit/${m}__seed$s" MODEL=$m EXPRESSION_ABLATION_ROOT=$(abl magic) SEED=$s
    done; done ;;
  seqonly|fullmodel_grouped)
    for sp in ${SPLITS:-event_sample1 gene_sample1 cell_sample1}; do
      python experiments/worm_ablations/preflight.py "$sp" > "logs/preflight_${sp}.log" 2>&1 || { echo "[skip] preflight failed for $sp (logs/preflight_${sp}.log)"; continue; }
      for s in ${SEEDS:-1}; do
        if [ "$1" = seqonly ]; then
          synth=1; [ "$sp" = rows_se ] || [ "$sp" = SE ] && synth=0
          submit "seqonly_${sp}_seed$s" "$RUNS/seqonly/${sp}__seed$s" MODEL=csn_seqonly SPLIT=$sp SEQONLY_SYNTH_SHAPES=$synth SEED=$s EXPRESSION_ABLATION_ROOT=$(abl magic)
        else
          submit "full_${sp}_seed$s" "$RUNS/fullmodel_grouped/${sp}__seed$s" MODEL=deltacellsplicenet SPLIT=$sp SEED=$s EXPRESSION_ABLATION_ROOT=$(abl magic)
        fi
      done
    done ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
