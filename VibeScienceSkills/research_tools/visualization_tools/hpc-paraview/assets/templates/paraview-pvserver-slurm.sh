#!/bin/bash
#SBATCH --job-name=paraview_server
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load paraview

srun -n "${SLURM_NTASKS}" pvserver --server-port=11111
