#!/bin/bash
set -euo pipefail

#BSUB -J hpc_job
#BSUB -n 1
#BSUB -W 01:00
#BSUB -o %J.out
#BSUB -e %J.err

echo "Job started at $(date)"
# module load ...
# mpirun <command>
echo "Job finished at $(date)"
