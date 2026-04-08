#!/bin/bash
#SBATCH --job-name=wrf_forecast
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=wrf_%j.out
#SBATCH --error=wrf_%j.err
#SBATCH --partition=compute

# Load modules
module load intel/2021.4
module load impi/2021.4
module load netcdf/4.8.1

# Set environment variables
export WRF_DIR=/path/to/WRF
export NETCDF=/path/to/netcdf
export LD_LIBRARY_PATH=$NETCDF/lib:$LD_LIBRARY_PATH

# Set OpenMP threads (for hybrid parallelization)
export OMP_NUM_THREADS=1
export KMP_AFFINITY=scatter

# Change to run directory
cd $SLURM_SUBMIT_DIR

# Check if met files exist
if [ ! -f "wrfinput_d01" ] || [ ! -f "wrfbdy_d01" ]; then
    echo "Error: wrfinput_d01 or wrfbdy_d01 not found"
    echo "Run real.exe first"
    exit 1
fi

# Run WRF
echo "Starting WRF at $(date)"
echo "Running on $(hostname)"
echo "Number of MPI tasks: $SLURM_NTASKS"

mpirun -np $SLURM_NTASKS $WRF_DIR/run/wrf.exe

echo "WRF completed at $(date)"

# Check for successful completion
if grep -q "SUCCESS COMPLETE WRF" rsl.out.0000; then
    echo "WRF run completed successfully"
else
    echo "WRF run may have failed. Check rsl files."
    exit 1
fi
