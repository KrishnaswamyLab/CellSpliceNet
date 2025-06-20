#!/bin/bash
# Job name
#SBATCH --job-name=build_graph_celltype

#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G

# Partition
#SBATCH --partition=pi_krishnaswamy

# Expected running time
#SBATCH --time=23:00:00

# Output and error files
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

# Load necessary modules
module load miniconda
conda activate env_3_8

# Run the Python script
python build_graph_celltype.py ../../data/expression/counts/241119b_scCounts_neuronsBulk_magic_filtered.csv ../../data/expression/mi_raw/
