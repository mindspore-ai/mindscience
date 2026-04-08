#!/bin/bash
#SBATCH --job-name=perf_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Timing baseline started at $(date)"
/usr/bin/time -v srun -n 1 -c "${SLURM_CPUS_PER_TASK}" ./run_case.sh
echo "Timing baseline finished at $(date)"
