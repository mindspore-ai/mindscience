#!/bin/bash
#SBATCH --job-name=gaussian_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

input_file="${1:-job.gjf}"
output_file="${2:-job.log}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# export g16root="/path/to/gaussian/root"
# . "$g16root/g16/bsd/g16.profile"
export GAUSS_SCRDIR="${TMPDIR:-/tmp}/${USER}/gaussian-${SLURM_JOB_ID}"
mkdir -p "$GAUSS_SCRDIR"

g16 < "$input_file" > "$output_file"
