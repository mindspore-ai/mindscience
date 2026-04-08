#!/bin/bash
set -euo pipefail

#SBATCH --job-name=hpc_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "Job started at $(date)"
# module load ...
# srun <command>
echo "Job finished at $(date)"
