#!/bin/bash
#SBATCH --job-name=cp2k_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=cp2k_%j.out
#SBATCH --error=cp2k_%j.err

module load cp2k/2024.1

# Set input and output files
INPUT="input.inp"
OUTPUT="output.out"

# Run CP2K
srun cp2k.psmp -i $INPUT -o $OUTPUT

echo "Job completed at $(date)"
