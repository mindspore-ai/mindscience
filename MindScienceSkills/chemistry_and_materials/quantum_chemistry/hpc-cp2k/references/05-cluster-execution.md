# CP2K Cluster Execution

## SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=cp2k
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=compute

module load cp2k/2024.1

srun cp2k.psmp -i input.inp -o output.out
```

## PBS Script

```bash
#!/bin/bash
#PBS -N cp2k_job
#PBS -l nodes=2:ppn=32
#PBS -l mem=128gb
#PBS -l walltime=48:00:00

module load cp2k/2024.1

mpirun -np 64 cp2k.psmp -i input.inp -o output.out
```

## Parallel Strategies

### MPI Parallelization
```bash
mpirun -np 64 cp2k.psmp -i input.inp
```

### MPI + OpenMP Hybrid Parallelization
```bash
export OMP_NUM_THREADS=4
mpirun -np 16 cp2k.psmp -i input.inp
```

### GPU Acceleration
```bash
cp2k.sdbg -i input.inp  # CUDA version
```

## Performance Optimization

| Parameter | Description |
|------|------|
| CUTOFF | Lower for speedup, but test convergence |
| FFT_LIB | Use FFTW or Intel MKL |
| MPI processes | Match CPU core count |
| OpenMP threads | Usually 2-4 |

## Memory Estimation

Memory requirement ≈ number of basis functions × cutoff energy × number of parallel processes

## Checkpoint Restart

```
&EXT_RESTART
  RESTART_FILE_NAME restart.wfn
&END EXT_RESTART
```