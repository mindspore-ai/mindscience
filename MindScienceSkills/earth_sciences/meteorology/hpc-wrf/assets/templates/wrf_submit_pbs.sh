#!/bin/bash
#PBS -N wrf_forecast
#PBS -l nodes=4:ppn=32
#PBS -l mem=128gb
#PBS -l walltime=24:00:00
#PBS -o wrf_$PBS_JOBID.out
#PBS -e wrf_$PBS_JOBID.err
#PBS -q batch

# Load modules
module load intel/2021.4
module load impi/2021.4
module load netcdf/4.8.1

# Set environment variables
export WRF_DIR=/path/to/WRF
export NETCDF=/path/to/netcdf
export LD_LIBRARY_PATH=$NETCDF/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1

# Change to run directory
cd $PBS_O_WORKDIR

# Create machine file
cat $PBS_NODEFILE > machines

# Run WRF
echo "Starting WRF at $(date)"
mpirun -machinefile machines -np 128 $WRF_DIR/run/wrf.exe
echo "WRF completed at $(date)"

# Check for success
if grep -q "SUCCESS COMPLETE WRF" rsl.out.0000; then
    echo "WRF run completed successfully"
else
    echo "WRF run may have failed"
    exit 1
fi
