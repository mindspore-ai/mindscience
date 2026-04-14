#!/bin/bash
#SBATCH --job-name=paraview_batch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

script_path="${1:-pvbatch_job.py}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load paraview

srun -n "${SLURM_NTASKS}" pvbatch "$script_path"