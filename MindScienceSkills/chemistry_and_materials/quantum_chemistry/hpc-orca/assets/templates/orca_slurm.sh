#!/bin/bash
#SBATCH --job-name=orca_job
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=orca_%j.out
#SBATCH --error=orca_%j.err

# Load modules
module load gcc/10.2.0 openmpi/4.0.4

# Set ORCA environment
export ORCADIR=/path/to/orca
export PATH=$ORCADIR:$PATH
export LD_LIBRARY_PATH=$ORCADIR:$LD_LIBRARY_PATH

# Create work directory
WRKDIR=/scratch/$USER/orca_$SLURM_JOB_ID
mkdir -p $WRKDIR
cd $WRKDIR

# Copy input files
cp $SLURM_SUBMIT_DIR/*.inp .
cp $SLURM_SUBMIT_DIR/*.xyz .

# Run ORCA
$ORCADIR/orca orca.inp >> orca.out

# Copy results back
cp -f $WRKDIR/* $SLURM_SUBMIT_DIR/

# Cleanup
cd $SLURM_SUBMIT_DIR
rm -rf $WRKDIR

echo "Job finished at $(date)"
