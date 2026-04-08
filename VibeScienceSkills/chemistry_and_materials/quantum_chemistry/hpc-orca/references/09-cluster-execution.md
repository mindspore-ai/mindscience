# Cluster Job Submission

## SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=orca_job
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G

module load gcc/10.2.0 openmpi/4.0.4

export ORCADIR=/path/to/orca
export PATH=$ORCADIR:$PATH
export LD_LIBRARY_PATH=$ORCADIR:$LD_LIBRARY_PATH

$ORCADIR/orca orca.inp >> orca.out
```

## PBS Script

```bash
#!/bin/bash
#PBS -N orca_job
#PBS -l nodes=1:ppn=8
#PBS -l walltime=24:00:00
#PBS -l mem=32gb

module load gcc openmpi

export ORCADIR=/path/to/orca
$ORCADIR/orca orca.inp >> orca.out
```

## Parallel Computing

```
%maxcore 4000
%pal nprocs 8 end
```

## Memory Estimation

- Total memory = nprocs × maxcore
- Example: 8 cores × 4GB = 32GB

## Checkpoint Restart

```
! B3LYP def2-SVP MOREAD

%moinp "old.gbw"
```

## Job Monitoring

```bash
# View queue
squeue -u $USER

# View progress
tail -f orca.out

# Check SCF
grep "SCF CONVERGED" orca.out
```