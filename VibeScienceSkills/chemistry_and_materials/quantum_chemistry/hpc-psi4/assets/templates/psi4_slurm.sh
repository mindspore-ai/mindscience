#!/bin/bash
#SBATCH --job-name=psi4_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=psi4_%j.out
#SBATCH --error=psi4_%j.err

module load psi4/1.9

# Run PSI4 (8 core parallel)
psi4 -n 8 -i input.py -o output.dat

echo "Job completed at $(date)"
