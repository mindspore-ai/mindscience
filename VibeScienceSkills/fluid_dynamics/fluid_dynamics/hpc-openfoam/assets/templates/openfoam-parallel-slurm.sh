#!/bin/bash
#SBATCH --job-name=openfoam_case
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

case_dir="${1:-$PWD}"
solver_or_module="${2:-simpleFoam}"

cd "$case_dir"

# module purge
# module load openfoam

decomposePar -force

if command -v "$solver_or_module" >/dev/null 2>&1; then
  run_cmd=("$solver_or_module" -parallel)
  log_name="$solver_or_module"
else
  run_cmd=(foamRun -solver "$solver_or_module" -parallel)
  log_name="foamRun_${solver_or_module}"
fi

srun -n "${SLURM_NTASKS}" "${run_cmd[@]}" > "log.${log_name}" 2>&1
reconstructPar
