#!/bin/bash
#SBATCH --job-name=feff_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --output=feff_%j.out
#SBATCH --error=feff_%j.err

module load feff/10.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run FEFF
feff10

echo "Job completed at $(date)"
