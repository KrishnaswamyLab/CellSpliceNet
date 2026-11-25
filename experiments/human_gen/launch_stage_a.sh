#!/bin/bash
# Base + tissue-delta test arms (2026-11-24), chained with Slurm dependencies. Run from the repo root.
#   stage A : csn_noexpr, PSI_HEAD=linear, bf16, 60k steps on gene_x_cell_em (event-mean targets), seeds 1, 2
#   emb     : extract_embeddings.py on the stage-A runs (best-validation checkpoint)
#   stage B : delta_head.py on the embeddings (gene_x_cell split)
#   arms    : track1_transfer.py exon arms, 5 tissue folds (TISSUE_FOLD 0-4) with feature/conv encoders
#             (RUN_TAG _arms) and with the CSN embeddings (RUN_TAG _armsemb, EMB_SEED=1)
#   report  : overnight_report.py when all of the above have ended
P=experiments/common/py.sbatch; G=experiments/common/gpu.sbatch
a1=$(SPLIT=gene_x_cell_em MODEL=csn_noexpr SEED=1 PSI_HEAD=linear AMP=bf16 N_STEPS=60000 TAG=_em sbatch --parsable -J hg_stageA experiments/common/train.sbatch)
a2=$(SPLIT=gene_x_cell_em MODEL=csn_noexpr SEED=2 PSI_HEAD=linear AMP=bf16 N_STEPS=60000 TAG=_em sbatch --parsable -J hg_stageA experiments/common/train.sbatch)
emb=$(PSI_HEAD=linear AMP=bf16 sbatch --parsable -J hg_emb --dependency=afterok:$a1:$a2 $G experiments/human_gen/extract_embeddings.py)
dh=$(sbatch --parsable -J hg_delta_head -c 4 --mem 64G -t 8:00:00 --dependency=afterok:$emb $G experiments/human_gen/delta_head.py)
deps="$a1:$a2:$emb:$dh"
for F in 0 1 2 3 4; do
  j=$(METHODS=exon_arms TISSUE_FOLD=$F RUN_TAG=_arms sbatch --parsable -J hg_track1 -c 4 --mem 160G -t 8:00:00 $G experiments/human_gen/track1_transfer.py)
  k=$(METHODS=exon_arms TISSUE_FOLD=$F RUN_TAG=_armsemb EXON_ENC=emb,emb+feat EMB_SEED=1 \
      sbatch --parsable -J hg_track1 -c 4 --mem 160G -t 8:00:00 --dependency=afterok:$emb $G experiments/human_gen/track1_transfer.py)
  deps="$deps:$j:$k"
done
TRAIN_LOGS="logs/hg_stageA_*.out" EMB_LOGS="logs/hg_emb_$emb.out" \
  sbatch -J hg_report -t 0:30:00 --dependency=afterany:$deps $P experiments/human_gen/overnight_report.py
echo "stage A $a1 $a2, emb $emb, delta head $dh"
