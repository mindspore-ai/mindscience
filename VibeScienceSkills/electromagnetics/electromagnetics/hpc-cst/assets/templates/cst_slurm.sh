#!/bin/bash
#==============================================================================
# CST Studio Suite SLURM Batch Script
# Purpose: Submit CST simulations to HPC cluster
#==============================================================================

#--- Job Configuration --------------------------------------------------------
#SBATCH --job-name=cst_simulation      # Job name
#SBATCH --output=cst_%j.out             # Standard output (%j = job ID)
#SBATCH --error=cst_%j.err              # Standard error

#--- Resource Requests --------------------------------------------------------
#SBATCH --nodes=2                       # Number of nodes
#SBATCH --ntasks-per-node=16            # MPI processes per node
#SBATCH --cpus-per-task=1               # CPUs per task
#SBATCH --mem=64G                       # Memory per node
#SBATCH --time=24:00:00                 # Wall time (HH:MM:SS)
#SBATCH --partition=compute             # Partition name

#--- Optional: GPU Request ----------------------------------------------------
# #SBATCH --gres=gpu:2                   # Request 2 GPUs (uncomment if needed)

#--- Optional: Job Array for Parameter Sweep ----------------------------------
# #SBATCH --array=1-10                   # Run 10 jobs (uncomment for sweep)

#==============================================================================

#--- Environment Setup --------------------------------------------------------
echo "=== Job Started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"

# Load CST module (adjust path as needed)
module load cst/2024
# Or set environment manually:
# export CST_ROOT=/opt/cst/2024
# export PATH=$CST_ROOT/bin:$PATH

# Set license server
export LM_LICENSE_FILE=27000@license.server

# Create working directory
WORK_DIR=/scratch/$USER/cst_$SLURM_JOB_ID
mkdir -p $WORK_DIR
cd $WORK_DIR

# Copy input files
cp $SLURM_SUBMIT_DIR/project.cst .
cp $SLURM_SUBMIT_DIR/*.cst . 2>/dev/null || true

#--- Parameter Sweep Setup (for job arrays) -----------------------------------
# Define parameter values for sweep
# params=(value1 value2 value3 value4 value5)
# param=${params[$SLURM_ARRAY_TASK_ID-1]}

#==============================================================================

#--- Run CST Simulation -------------------------------------------------------
echo "=== Starting CST Simulation ==="

# Basic run (adjust project name)
# cst_designer -m -r project.cst

# MPI parallel run
cst_designer -m -r -withmpi $SLURM_NTASKS project.cst

# Hybrid MPI + SMP run
# cst_designer -m -r -withmpi $SLURM_JOB_NUM_NODES -withsmp $SLURM_NTASKS_PER_NODE project.cst

# GPU acceleration
# cst_designer -m -r -withgpu 2 project.cst

# With parameters
# cst_designer -m -r -p "freq=2.4e9" -p "length=50" project.cst

#--- Check Exit Status --------------------------------------------------------
if [ $? -eq 0 ]; then
    echo "=== CST Simulation Completed Successfully ==="
else
    echo "=== CST Simulation Failed ==="
    exit 1
fi

#==============================================================================

#--- Post-Processing ----------------------------------------------------------
echo "=== Post-Processing ==="

# Export results (adjust as needed)
# cst_designer -m -e -export "S-parameters" project.cst
# cst_designer -m -e -export "farfield" project.cst

#--- Copy Results Back --------------------------------------------------------
echo "=== Copying Results ==="
mkdir -p $SLURM_SUBMIT_DIR/results_$SLURM_JOB_ID
cp -r *.res $SLURM_SUBMIT_DIR/results_$SLURM_JOB_ID/ 2>/dev/null || true
cp -r Result $SLURM_SUBMIT_DIR/results_$SLURM_JOB_ID/ 2>/dev/null || true
cp *.log $SLURM_SUBMIT_DIR/results_$SLURM_JOB_ID/ 2>/dev/null || true

#--- Cleanup ------------------------------------------------------------------
cd $SLURM_SUBMIT_DIR
rm -rf $WORK_DIR

echo "=== Job Finished at $(date) ==="
