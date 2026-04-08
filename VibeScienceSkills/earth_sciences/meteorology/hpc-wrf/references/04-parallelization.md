# WRF Parallelization

## Parallelization Models

### MPI (Distributed Memory)
```bash
# Configure for MPI
./configure  # Choose dmpar option

# Run with MPI
mpirun -np 128 ./wrf.exe
```

### OpenMP (Shared Memory)
```bash
# Configure for OpenMP
./configure  # Choose smpar option

# Run with OpenMP
export OMP_NUM_THREADS=32
./wrf.exe
```

### Hybrid (MPI + OpenMP)
```bash
# Configure for hybrid
./configure  # Choose dm+sm option

# Run hybrid
export OMP_NUM_THREADS=4
mpirun -np 32 ./wrf.exe
```

## Domain Decomposition

### Automatic Decomposition
WRF automatically decomposes domain based on MPI tasks.

### Manual Decomposition
```
&domains
 nproc_x = 8,              ! Processors in x
 nproc_y = 16,             ! Processors in y
/
```

## I/O Optimization

### Quilting (Asynchronous I/O)
```
&namelist_quilt
 nio_tasks_per_group = 4,  ! I/O tasks per group
 nio_groups = 1,           ! Number of I/O groups
/
```

Benefits:
- Overlaps computation and I/O
- Reduces I/O bottleneck
- Recommended for large domains

### History Output Options
```
&time_control
 history_interval = 180,   ! Output interval (min)
 frames_per_outfile = 1,    ! Frames per file
 io_form_history = 2,       ! NetCDF format
/
```

## Performance Tuning

### Memory Considerations
- Each MPI task needs ~1-2 GB memory
- Check memory with `ulimit -a`

### CPU Affinity
```bash
# Intel MPI
export I_MPI_PIN_DOMAIN=omp

# OpenMPI
mpirun --bind-to core -np 128 ./wrf.exe
```

### Recommended Configurations

| Domain Size | MPI Tasks | OpenMP Threads | Total Cores |
|-------------|-----------|----------------|-------------|
| 100x100 | 16 | 1 | 16 |
| 300x300 | 64 | 2 | 128 |
| 500x500 | 128 | 2 | 256 |
| 1000x1000 | 256 | 4 | 1024 |
