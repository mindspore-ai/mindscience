#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

case_dir=$(printf "cases/case_%03d" "${SLURM_ARRAY_TASK_ID}")
cd "$case_dir"

echo "Starting array index ${SLURM_ARRAY_TASK_ID} at $(date)"
srun -n 1 ./run_case.sh
echo "Finished array index ${SLURM_ARRAY_TASK_ID} at $(date)"
