#!/usr/bin/env bash
#$ -cwd
#$ -V
#$ -j y
#$ -l mem_free=16G

echo $1 $2

source activate scrna

./score_fisher.py /home/ingn/KnottLab/bladder/BladderGEX_run2_working.h5ad \
  /home/ingn/KnottLab/bladder/fisher_interactions/receptor_genelists_2020_06_30_scores.h5ad \
  /home/ingn/KnottLab/bladder/fisher_interactions/predicted_signaling \
  -s $1 -r $2 \
  --col SubType_v2 \
  --sample_col Patient \
  --receptor_ligand_dict /home/ingn/KnottLab/bladder/fisher_interactions/cabello_aguilar_dict.pkl \
  --reps 5000
