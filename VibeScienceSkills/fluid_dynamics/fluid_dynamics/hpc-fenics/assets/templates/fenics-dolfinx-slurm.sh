#!/bin/bash
#SBATCH --job-name=fenics_case
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

script_path="${1:-poisson_dolfinx.py}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# module purge
# module load python
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate dolfinx-env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun -n "${SLURM_NTASKS}" python "$script_path"
