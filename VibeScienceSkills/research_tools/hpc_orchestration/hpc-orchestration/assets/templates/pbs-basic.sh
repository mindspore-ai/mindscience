#!/bin/bash
set -euo pipefail

#PBS -N hpc_job
#PBS -l select=1:ncpus=1
#PBS -l walltime=01:00:00
#PBS -o hpc_job.out
#PBS -e hpc_job.err

cd $PBS_O_WORKDIR
echo "Job started at $(date)"
# module load ...
# mpirun <command>
echo "Job finished at $(date)"
