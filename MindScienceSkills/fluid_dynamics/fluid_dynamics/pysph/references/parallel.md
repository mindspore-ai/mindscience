# Parallel Execution with PySPH

PySPH supports multiple levels of parallelism for high-performance simulations.

## OpenMP Parallelism

Multi-threaded execution on shared-memory systems.

**Enable**:
```bash
pysph run dam_break_2d --backend=cython --threads=8
```

**Characteristics**:
- Shared memory parallelism
- Automatic load balancing
- Good for multi-core workstations
- Limited to single node

**Configuration**:
```python
# In application setup
app = Application(
    particle_arrays=[particles],
    nnps_factory=lambda: nps,
    integrator=integrator,
    equations=equations,
    backend='cython',  # Cython backend
    n_threads=8         # Number of OpenMP threads
)
```

## OpenCL/GPU Parallelism

Execute on GPU devices for massive parallelism.

**Enable**:
```bash
pysph run dam_break_2d --backend=opencl
```

**Characteristics**:
- Massive parallelism (thousands of threads)
- Good for large particle counts
- Requires OpenCL-compatible GPU
- May have overhead for small problems

**Configuration**:
```python
from pysph.base.opencl import get_context

# Select GPU device
ctx = get_context(device_type='GPU')

app = Application(
    particle_arrays=[particles],
    nnps_factory=lambda: nps,
    integrator=integrator,
    equations=equations,
    backend='opencl',    # OpenCL backend
    opencl_context=ctx    # OpenCL context
)
```

**Device selection**:
```python
# List available devices
from pysph.base.opencl import get_device
devices = get_device()

# Select specific device
ctx = get_context(device_id=0)  # First GPU
```

## MPI Parallelism

Distributed memory parallelism across multiple nodes.

**Enable**:
```bash
mpirun -np 4 pysph run dam_break_2d --backend=mpich
```

**Characteristics**:
- Distributed memory parallelism
- Scalable to many nodes
- Requires MPI installation
- Automatic domain decomposition

**Configuration**:
```python
from pysph.parallel import ZoltanParallelManagerGeometric

# Setup MPI communicator
from mpi4py import MPI
comm = MPI.COMM_WORLD

# Create parallel manager
pm = ZoltanParallelManagerGeometric(
    dim=2,
    particles=[particles],
    comm=comm,
    radius_scale=3.0,
    lb_method='rcb'  # Recursive Coordinate Bisection
)

app = Application(
    particle_arrays=[particles],
    nnps_factory=lambda: pm,
    integrator=integrator,
    equations=equations,
    backend='mpich',      # MPI backend
    parallel_manager=pm
)
```

## Load Balancing Methods

PySPH with Zoltan supports multiple load balancing algorithms:

### Recursive Coordinate Bisection (RCB)

```python
pm = ZoltanParallelManagerGeometric(
    dim=2,
    particles=[particles],
    comm=comm,
    radius_scale=3.0,
    lb_method='rcb'
)
```

**Characteristics**:
- Fast decomposition
- Good for uniform distributions
- May create imbalanced loads for clustered particles

### Recursive Inertial Bisection (RIB)

```python
pm = ZoltanParallelManagerGeometric(
    dim=2,
    particles=[particles],
    comm=comm,
    radius_scale=3.0,
    lb_method='rib'
)
```

**Characteristics**:
- Considers particle inertia
- Better for non-uniform distributions
- Slower than RCB
- Good for problems with varying density

### Hilbert Space Filling Curves (HSFC)

```python
pm = ZoltanParallelManagerGeometric(
    dim=2,
    particles=[particles],
    comm=comm,
    radius_scale=3.0,
    lb_method='hsfc'
)
```

**Characteristics**:
- Preserves spatial locality
- Best for communication patterns
- More complex decomposition
- Good for hierarchical architectures

## Load Balancing Selection Guide

| Problem Type | Recommended Method | Reason |
|--------------|-------------------|---------|
| Uniform particle distribution | RCB | Fast, simple |
| Clustered particles | RIB | Considers inertia |
| Communication-intensive | HSFC | Spatial locality |
| Dynamic particle motion | RIB or HSFC | Better adaptation |
| Large-scale simulation | HSFC | Best scalability |

## Parallel Neighbor Search

### Local Neighbor Search

For OpenMP and OpenCL backends:

```python
from pysph.base.nnps import LinkedListNNPS

nps = LinkedListNNPS(
    dim=2,
    particles=[particles],
    radius_scale=3.0
)
```

**Characteristics**:
- Thread-local neighbor lists
- No communication overhead
- Good for shared-memory systems

### Distributed Neighbor Search

For MPI backend with Zoltan:

```python
from pysph.parallel import ZoltanParallelManagerGeometric

pm = ZoltanParallelManagerGeometric(
    dim=2,
    particles=[particles],
    comm=comm,
    radius_scale=3.0
)
```

**Characteristics**:
- Automatic ghost particle creation
- Distributed neighbor lists
- Communication for ghost updates
- Domain decomposition

## Particle Tags in Parallel Execution

PySPH uses tags to identify particle types:

```python
class ParticleTAGS:
    Local = 0   # Real particles owned by this processor
    Remote = 1  # Real particles owned by other processors
    Ghost = 2   # Ghost particles for boundary conditions
```

**Usage**:
```python
# Only operate on local particles
Group(equations=[Eq1(...)], real=True)

# Include ghost particles
Group(equations=[Eq2(...)], real=False)
```

## Global ID and Processor ID

Each particle has unique identifiers:

```python
# Global unique identifier
pa.gid  # Unique across all processors

# Processor ID
pa.pid  # Current processor owner
```

**Use cases**:
- Global particle tracking
- Load balancing statistics
- Debugging particle distribution
- Custom parallel algorithms

## Communication Patterns

### Ghost Particle Exchange

Automatic in MPI mode:
1. Domain decomposition
2. Ghost particle creation
3. Position/property exchange
4. Neighbor search update

### Reduction Operations

Global reductions across processors:

```python
class GlobalStats(Equation):
    def reduce(self, dst, t, dt):
        m = serial_reduce_array(dst.m, 'sum')
        dst.total_mass[0] = parallel_reduce_array(m, 'sum')
```

**Operations**: `sum`, `prod`, `max`, `min`

### Custom Communication

For specialized communication patterns:

```python
def my_communication(dst, t, dt):
    # Custom MPI communication
    if dst.pid == 0:
        # Root processor gathers data
        pass

equations = [
    Group(
        equations=[MyEquation(...)],
        post=my_communication
    )
]
```

## Performance Optimization

### Minimize Communication

1. **Use local operations**: Prefer serial_reduce_array over parallel_reduce_array
2. **Batch communications**: Group multiple messages
3. **Overlap computation and communication**: Non-blocking operations
4. **Choose appropriate load balancer**: Match problem characteristics

### Memory Management

1. **Limit ghost particles**: Only create necessary ghosts
2. **Reuse buffers**: Avoid repeated allocations
3. **Compact arrays**: Remove unused particles
4. **Use appropriate data types**: Minimize memory footprint

### Load Balancing

1. **Monitor load imbalance**: Check particle distribution
2. **Rebalance periodically**: Dynamic load balancing
3. **Choose appropriate method**: RCB vs RIB vs HSFC
4. **Consider problem characteristics**: Spatial locality, particle clustering

## Scalability Considerations

### Strong Scaling

Fixed problem size, increasing processors:

**Factors affecting scalability**:
- Communication overhead
- Load imbalance
- Ghost particle ratio
- Surface-to-volume ratio

**Typical scaling**:
- OpenMP: Good to 8-16 cores
- OpenCL: Good to thousands of threads
- MPI: Good to hundreds of nodes

### Weak Scaling

Fixed problem size per processor, increasing processors:

**Factors affecting scalability**:
- Global operations
- Reduction frequency
- Synchronization points
- I/O operations

## Debugging Parallel Code

1. **Check particle distribution**: Verify load balance
2. **Monitor ghost particles**: Ensure correct creation
3. **Profile communication**: Identify bottlenecks
4. **Test with different backends**: Compare performance
5. **Verify results**: Check for parallel artifacts

## Common Issues and Solutions

### Load Imbalance

**Problem**: Some processors have more work

**Solutions**:
- Change load balancing method
- Increase rebalancing frequency
- Use HSFC for better spatial locality

### Communication Overhead

**Problem**: Too much time spent communicating

**Solutions**:
- Reduce ghost particle count
- Batch communications
- Use non-blocking operations
- Overlap computation and communication

### Deadlock

**Problem**: Processors waiting indefinitely

**Solutions**:
- Check MPI communication patterns
- Ensure consistent message ordering
- Verify reduction operations
- Use proper synchronization

### Incorrect Results

**Problem**: Parallel results differ from serial

**Solutions**:
- Check ghost particle updates
- Verify reduction operations
- Ensure consistent initial conditions
- Validate equation implementation

## Hybrid Parallelism

Combine multiple parallelism levels:

```python
# MPI + OpenMP
mpirun -np 4 pysph run dam_break_2d --backend=mpich --threads=8

# MPI + OpenCL
mpirun -np 4 pysph run dam_break_2d --backend=mpich --opencl
```

**Configuration**:
- MPI for inter-node communication
- OpenMP for intra-node parallelism
- OpenCL for intra-node GPU acceleration

## Best Practices

1. **Start with serial**: Verify correctness before parallelizing
2. **Profile first**: Identify bottlenecks before optimization
3. **Use appropriate backend**: Match problem and hardware
4. **Monitor performance**: Track scaling and efficiency
5. **Test scalability**: Verify strong and weak scaling
6. **Document configuration**: Record optimal settings
