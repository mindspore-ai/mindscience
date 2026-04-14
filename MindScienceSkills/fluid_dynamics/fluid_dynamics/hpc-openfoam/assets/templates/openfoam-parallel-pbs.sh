#!/bin/bash
#PBS -N openfoam_case
#PBS -l nodes=1:ppn=16
#PBS -l walltime=02:00:00
#PBS -o pbs-%J.out
#PBS -e pbs-%J.err

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

mpirun -np "${PBS_NUM_PPN:-16}" "${run_cmd[@]}" > "log.${log_name}" 2>&1
reconstructPar
