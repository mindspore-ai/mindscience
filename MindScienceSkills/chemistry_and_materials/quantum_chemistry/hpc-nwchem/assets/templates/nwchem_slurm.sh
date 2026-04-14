#!/bin/bash
#SBATCH --job-name=nwchem_job
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --output=nwchem_%j.out
#SBATCH --error=nwchem_%j.err

module load nwchem/7.2

# Set parallel environment
export OMP_NUM_THREADS=1

# Run NWChem
mpirun -np $SLURM_NTASKS nwchem input.nw > output.out

echo "Job completed at $(date)"
