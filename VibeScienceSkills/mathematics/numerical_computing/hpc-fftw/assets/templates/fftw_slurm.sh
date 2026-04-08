#!/bin/bash
#SBATCH --job-name=fftw_benchmark
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=fftw_%j.out
#SBATCH --error=fftw_%j.err

module load fftw/3.3.10

# For MPI FFTW
export OMP_NUM_THREADS=1

# Run MPI FFTW benchmark
mpirun -np 64 ./mpi_3d_dft

# For OpenMP FFTW
# export OMP_NUM_THREADS=32
# ./openmp_fft
