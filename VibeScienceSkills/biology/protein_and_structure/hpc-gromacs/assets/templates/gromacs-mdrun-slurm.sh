#!/bin/bash
#SBATCH --job-name=gromacs_md
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

deffnm="${1:-md}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# module purge
# module load gromacs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun -n "${SLURM_NTASKS}" gmx_mpi mdrun -deffnm "$deffnm"
