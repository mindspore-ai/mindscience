#!/bin/bash
#SBATCH --job-name=lammps_case
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

input_script="${1:-in.case}"
log_file="${2:-log.lammps}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# module purge
# module load lammps

srun -n "${SLURM_NTASKS}" lmp -in "$input_script" -log "$log_file"
