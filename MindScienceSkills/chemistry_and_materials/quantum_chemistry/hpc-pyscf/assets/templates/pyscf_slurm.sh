#!/bin/bash
#SBATCH --job-name=pyscf_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=pyscf_%j.out
#SBATCH --error=pyscf_%j.err

module load python/3.10
module load pyscf/2.5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python calculation.py

echo "Job completed at $(date)"
