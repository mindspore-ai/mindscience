#!/bin/bash
#SBATCH --job-name=openblas_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=openblas_%j.out
#SBATCH --error=openblas_%j.err

module load openblas/0.3.21

# Set number of threads
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32

# Run benchmark
./blas_level3

# For single-threaded comparison
# export OPENBLAS_NUM_THREADS=1
# ./blas_level3
