#!/bin/bash
#SBATCH --job-name=qe_job
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

input_file="${1:-scf.in}"
output_file="${2:-pw.out}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# module purge
# module load quantum-espresso

export OMP_NUM_THREADS=1
srun -n "${SLURM_NTASKS}" pw.x -in "$input_file" > "$output_file" 2>&1
