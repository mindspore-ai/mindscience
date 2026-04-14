#!/bin/bash
#SBATCH --job-name=packed_jobs
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

mkdir -p logs

echo "Packed job started at $(date)"

srun --exclusive -N 1 -n 1 ./task_a.sh > logs/task_a-${SLURM_JOB_ID}.log 2>&1 &
pid_a=$!

srun --exclusive -N 1 -n 1 ./task_b.sh > logs/task_b-${SLURM_JOB_ID}.log 2>&1 &
pid_b=$!

srun --exclusive -N 1 -n 1 ./task_c.sh > logs/task_c-${SLURM_JOB_ID}.log 2>&1 &
pid_c=$!

srun --exclusive -N 1 -n 1 ./task_d.sh > logs/task_d-${SLURM_JOB_ID}.log 2>&1 &
pid_d=$!

wait "$pid_a"
wait "$pid_b"
wait "$pid_c"
wait "$pid_d"

echo "Packed job finished at $(date)"
