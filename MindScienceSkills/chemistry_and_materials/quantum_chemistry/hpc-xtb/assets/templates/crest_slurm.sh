#!/bin/bash
#SBATCH --job-name=crest_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=compute
#SBATCH --output=crest_%j.out
#SBATCH --error=crest_%j.err

module load xtb/6.6
module load crest/2.12

export OMP_NUM_THREADS=$SLURM_NTASKS_PER_NODE

# Conformer search
crest molecule.xyz --gfn2 --gbsa water -T $OMP_NUM_THREADS

echo "Conformer search completed at $(date)"
