# Parallel Computing in gprMax

gprMax supports parallel computing for accelerated simulations.

## OpenMP Parallelization

### Basic OpenMP Usage

```bash
# OpenMP is automatically detected and used
python -m gprMax input_file.in
```

OpenMP provides shared-memory parallelization for CPU-based acceleration.

## MPI Parallelization

### Basic MPI Usage

```bash
# MPI task farm (master + workers)
python -m gprMax input_file.in -mpi 61

# MPI without spawn mechanism
python -m gprMax input_file.in -mpi 61 --mpi-no-spawn
```

MPI task farm supports distributed memory and parallel execution across multiple nodes.

### MPI Configuration

```python
# MPI configuration in input file
#mpi: 61  # master + workers
#task: 0  # task identifier for HPC/Grider Engine
```

## GPU Acceleration

### Basic GPU Usage

```bash
# Use NVIDIA GPU
python -m gprMax input_file.in -gpu 0

# Use specific GPU
python -m gprMax input_file.in -gpu 0 1
```

GPU acceleration provides significant speedup for large simulations.

### GPU Configuration

```python
# GPU configuration in input file
#gpu: 0 1  # GPU device ID(s)
```

## Hybrid Parallelization

### OpenMP + GPU

```bash
# OpenMP with GPU acceleration
python -m gprMax input_file.in -gpu 0
```

### MPI + GPU

```bash
# MPI with GPU acceleration
python -m gprMax input_file.in -mpi 61 -gpu 0
```

## Performance Considerations

### Memory Requirements

- **OpenMP**: Requires sufficient shared memory
- **MPI**: Each process needs enough memory for local domain
- **GPU**: Requires sufficient GPU memory

### Communication Overhead

- **OpenMP**: Low communication overhead
- **MPI**: Higher communication overhead
- **GPU**: Minimal communication overhead

### Scalability

- **OpenMP**: Good for shared-memory systems
- **MPI**: Good for distributed-memory systems
- **GPU**: Best for compute-bound problems

## Best Practices

### Parallel Strategy Selection

1. **Small simulations**: Use serial execution
2. **Medium simulations**: Use OpenMP parallelization
3. **Large simulations**: Use MPI task farm
4. **GPU-bound problems**: Use GPU acceleration

### Resource Optimization

1. **Balance memory**: Allocate sufficient memory per process
2. **Optimize domain decomposition**: Minimize communication
- **Consider hybrid approaches**: Combine OpenMP + GPU

### Performance Tuning

1. **Benchmark different configurations**: Test OpenMP, MPI, GPU
2. **Profile memory usage**: Monitor memory consumption
3. **Optimize domain size**: Balance accuracy and speed
4. **Monitor communication**: Track MPI communication overhead

## Troubleshooting

### OpenMP Issues

**Problem**: OpenMP not detected or fails

**Solutions**:
1. Verify OpenMP is installed
2. Check number of threads available
3. Reduce memory per thread
4. Check for OpenMP compatibility issues

### MPI Issues

**Problem**: MPI processes fail to communicate

**Solutions**:
1. Verify MPI is installed
2. Check network connectivity
3. Check task farm configuration
4. Reduce communication frequency

### GPU Issues

**Problem**: GPU acceleration fails or is slow

**Solutions**:
1. Verify CUDA is installed
2. Check GPU device is available
3. Check GPU memory is sufficient
4. Reduce problem size or resolution

### Performance Issues

**Problem**: Parallelization doesn't improve performance

**Solutions**:
1. Profile simulation to find bottlenecks
2. Check for load imbalance
3. Optimize domain decomposition
4. Consider different parallel strategy