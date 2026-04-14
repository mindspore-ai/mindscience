#!/bin/bash
#SBATCH --job-name=su2_case
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

cfg_file="${1:-case.cfg}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# module purge
# Load SU2 — replace with the site-specific module name if different:
# module load su2

srun -n "${SLURM_NTASKS}" SU2_CFD "$cfg_file" > log.su2 2>&1
