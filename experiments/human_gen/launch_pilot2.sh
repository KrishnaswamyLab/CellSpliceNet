#!/bin/bash
# Pilot 2 (2026-11-23): PSI_HEAD=twohead (exon mean + tissue deviation, raw-PSI loss), bf16, 8-tissue splits.
# Two model families, matched to each other:
#   cues    : deltacellsplicenet | csn_seqonly | csn_nostruct | csn_noexpr          (annotations + ROI kept)
#   nocues  : csn_full_nocues_strict | csn_seqonly_nocues | csn_full_nocues_nostruct | csn_full_nocues_noexpr
#             (annotations collapsed, ROI zeroed: the rebuttal Minor 3 SeqOnly definition)
# + loss-only control: deltacellsplicenet PSI_HEAD=linear.  Each run gets its own afterok eval job
# (1,500 test events, final + best checkpoints). Then: split_baselines.py / tissue_baselines.py, pilot_report.py.
# Run from the repo root.
CUES="deltacellsplicenet csn_seqonly csn_nostruct csn_noexpr"
NOCUES="csn_full_nocues_strict csn_seqonly_nocues csn_full_nocues_nostruct csn_full_nocues_noexpr"
sub() {  # split model seed head tag
  local S=$1 M=$2 SEED=$3 HEAD=$4 TAG=$5
  local j=$(SPLIT=$S MODEL=$M SEED=$SEED PSI_HEAD=$HEAD TAG=$TAG AMP=bf16 N_STEPS=80000 TIME_BUDGET_S=41400 \
            sbatch --parsable -J hg_p2_${S}_${M}${TAG}_s${SEED} experiments/common/train.sbatch)
  local e=$(CFG=$S PATTERN=hg_${S}_${M}${TAG}_s${SEED} CKPT_NAMES="final_model best_validation_model" AMP=bf16 MAX_EVENTS=1500 \
            sbatch --parsable -J hg_p2eval_${S}_${M}${TAG}_s${SEED} --time=4:00:00 --dependency=afterok:$j experiments/common/eval.sbatch)
  echo "$S $M$TAG s$SEED train=$j eval=$e"
}
for SEED in 1 2; do for M in $CUES $NOCUES; do sub gene_k8 $M $SEED twohead _2h; done; done
sub gene_k8 deltacellsplicenet 1 linear _lin
for M in deltacellsplicenet csn_seqonly csn_full_nocues_strict csn_seqonly_nocues; do sub chrom_k8 $M 1 twohead _2h; done
for SEED in 1 2; do sub cell_k8 deltacellsplicenet $SEED twohead _2h; done
