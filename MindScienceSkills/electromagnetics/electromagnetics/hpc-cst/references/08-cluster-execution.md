# Cluster Execution

## HPC Execution Overview

CST Studio Suite supports parallel execution on HPC clusters:

| Mode | Description |
|------|-------------|
| MPI | Distributed memory parallelism |
| SMP | Shared memory parallelism |
| Hybrid | MPI + SMP combination |
| GPU | Graphics processing unit acceleration |

## Job Scheduler Integration

### SLURM

```bash
#!/bin/bash
#SBATCH --job-name=cst_simulation
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=compute

# Load CST module
module load cst/2024

# Set CST environment
export CST_ROOT=/opt/cst/2024
export LM_LICENSE_FILE=27000@license.server

# Run CST
cst_designer -m -r -withmpi 32 project.cst
```

### PBS/Torque

```bash
#!/bin/bash
#PBS -N cst_simulation
#PBS -l nodes=2:ppn=16
#PBS -l mem=64gb
#PBS -l walltime=24:00:00
#PBS -q compute

# Load CST module
module load cst/2024

# Run CST
cst_designer -m -r -withmpi 32 project.cst
```

### SGE (Sun Grid Engine)

```bash
#!/bin/bash
#$ -N cst_simulation
#$ -pe mpi 32
#$ -l mem=64G
#$ -l h_rt=24:00:00
#$ -q compute.q

# Load CST module
module load cst/2024

# Run CST
cst_designer -m -r -withmpi 32 project.cst
```

## CST Command Line Options

### Basic Options

| Option | Description |
|--------|-------------|
| `-m` | Run without GUI (batch mode) |
| `-r` | Run simulation |
| `-i` | Import file |
| `-e` | Export results |
| `-p` | Set parameter |

### Parallel Options

| Option | Description |
|--------|-------------|
| `-withmpi N` | Use N MPI processes |
| `-withsmp N` | Use N SMP threads |
| `-withgpu N` | Use N GPUs |

### Example Commands

```bash
# Single simulation
cst_designer -m -r project.cst

# MPI parallel (32 cores)
cst_designer -m -r -withmpi 32 project.cst

# Hybrid (4 nodes × 8 cores)
cst_designer -m -r -withmpi 4 -withsmp 8 project.cst

# GPU acceleration
cst_designer -m -r -withgpu 2 project.cst

# With parameters
cst_designer -m -r -p "freq=2.4e9" -p "length=50" project.cst
```

## Parameter Sweep on Cluster

### Sequential Sweep

```bash
# Run parameter sweep sequentially
for length in 40 45 50 55 60; do
    cst_designer -m -r -p "length=${length}" project.cst
done
```

### Parallel Sweep (Job Array)

```bash
#!/bin/bash
#SBATCH --job-name=cst_sweep
#SBATCH --array=1-5
#SBATCH --ntasks-per-node=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Parameter values
lengths=(40 45 50 55 60)
length=${lengths[$SLURM_ARRAY_TASK_ID-1]}

# Run CST
cst_designer -m -r -p "length=${length}" project.cst
```

## Checkpoint and Restart

### Enable Checkpoint

```
# In CST project settings
Checkpoint: Enabled
Interval: 1 hour
Directory: /scratch/checkpoints/
```

### Restart from Checkpoint

```bash
# Restart from last checkpoint
cst_designer -m -r -restart project.cst
```

## Resource Management

### Memory Estimation

| Solver | Memory per cell |
|--------|-----------------|
| Transient | ~100 bytes |
| Frequency Domain | ~500 bytes |
| Integral Equation | ~1 KB |

### Example Estimation

```
Mesh cells: 10 million
Solver: Transient
Memory: 10M × 100 bytes = 1 GB + overhead ≈ 2 GB
```

### GPU Memory

```
GPU memory: 16 GB
Maximum cells: ~100 million (Transient)
```

## Performance Optimization

### Mesh Optimization

1. Use appropriate mesh density
2. Enable adaptive mesh refinement
3. Use symmetry when possible

### Solver Selection

| Problem Size | Recommended Solver |
|--------------|-------------------|
| Small (< 1M cells) | Frequency Domain |
| Medium (1-10M cells) | Transient |
| Large (> 10M cells) | Transient + MPI |

### Parallel Scaling

| Cores | Speedup | Efficiency |
|-------|---------|------------|
| 1 | 1× | 100% |
| 8 | 7× | 87% |
| 32 | 25× | 78% |
| 128 | 80× | 62% |

## Best Practices

1. **Estimate resources** before submission
2. **Use checkpoint** for long jobs
3. **Monitor job progress** via log files
4. **Clean up** temporary files
5. **Archive results** after completion

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| License error | No license available | Check license server |
| Memory overflow | Insufficient memory | Reduce mesh or increase memory |
| Timeout | Walltime exceeded | Increase walltime or use checkpoint |
| MPI error | Network issues | Check MPI configuration |
| GPU error | Driver mismatch | Update GPU driver |
