#!/bin/bash
#SBATCH --job-name=apptainer_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module purge
# module load apptainer

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
scratch_dir="${TMPDIR:-/tmp}/${USER}/job-${SLURM_JOB_ID}"
mkdir -p "$scratch_dir"

echo "Container job started at $(date)"
srun -n 1 apptainer exec \
  --bind "$PWD:/work" \
  --bind "${scratch_dir}:/scratch" \
  image.sif \
  bash -lc 'cd /work && ./run_inside_container.sh'
echo "Container job finished at $(date)"
