# Parallel Execution

OpenFOAM provides comprehensive parallel execution capabilities for large-scale simulations.

## Domain Decomposition

### scotch Method

**Description**: Recursive coordinate bisection

**Configuration**:
```python
decomposeDict = {
    'method': 'scotch',
    'n': 4  # Number of subdomains
}

# Run with scotch decomposition
mpirun -np 4 blockMeshDict cavity -parallel
```

**Characteristics**:
- Fast decomposition
- Good load balancing
- Minimal communication overhead
- Good for uniform particle distributions

### hierarchical Method

**Description**: Hierarchical decomposition

**Configuration**:
```python
decomposeDict = {
    'method': 'hierarchical',
    'n': 4
}

# Run with hierarchical decomposition
mpirun -np 4 blockMeshDict cavity -parallel
```

**Characteristics**:
- Preserves spatial locality
- Better for clustered distributions
- More complex decomposition
- Slower than scotch

### manual Method

**Description**: Manual decomposition

**Configuration**:
```python
decomposeDict = {
    'method': 'manual',
    'subDicts': {
        'region0': {
            'faces': 'f(xMin yMin yMax)(xMax yMax)',
            'owner': 'none'
        },
        'region1': {
            'faces': 'f(xMin yMin yMax)(xMax yMax)',
            'owner': 'none'
        }
    }
}
```

**Characteristics**:
- Complete control over decomposition
- Good for custom partitioning
- Requires manual specification

## Load Balancing

### Dynamic Load Balancing

**Description**: Automatic load balancing during simulation

**Configuration**:
```python
# Enable dynamic load balancing
applicationDict = {
    'application': 'compressibleInterFoam',
    'loadBalancing': True
}
```

**Characteristics**:
- Automatic redistribution
- Reduces load imbalance
- May cause communication overhead

### Load Balancing Methods

| Method | Speed | Use Case |
|--------|-------|---------|
| scotch | Fast | Uniform distributions |
| hierarchical | Medium | Clustered distributions |
| manual | Slow | Custom control |

## Parallel Solvers

### SIMPLE Solver Parallelization

**Configuration**:
```python
applicationDict = {
    'solver': 'simpleFoam',
    'maxCo': 0.1,
    'maxAlphaCo': 0.1
}
```

**Characteristics**:
- SIMPLE solver supports parallel execution
- Automatic domain decomposition
- Pressure solver parallelized

### PIMPLE Solver Parallelization

**Configuration**:
```python
applicationDict = {
    'solver': 'pimpleFoam',
    'maxCo': 0.7,
    'maxAlphaCo': 0.3
}
```

**Characteristics**:
- PIMPLE solver supports parallel execution
- Better convergence properties
- Higher computational cost

### PISO Solver Parallelization

**Configuration**:
```python
applicationDict = {
    'solver': 'PISO',
    'maxCo': 0.8,
    'maxAlphaCo': 0.2
}
```

**Characteristics**:
- PISO solver supports parallel execution
- Pressure-implicit parallelization
- Good for multiphase flows

## Parallel Execution

### Basic Parallel Execution

```bash
# Run with 4 cores
mpirun -np 4 blockMeshDict cavity

# Run with specific method
mpirun -np 4 blockMeshDict cavity -parallel -decompose metis
```

### Hybrid Parallelism

```bash
# MPI + threads
mpirun -np 4 -npernode 2 blockMeshDictator -parallel

# MPI + OpenMP
mpirun -np 4 -npernode 2 blockMeshDictor -parallel -threads 4
```

## Parallel Mesh Generation

### Decomposed Mesh Generation

```python
decomposeDict = {
    'method': 'scotch',
    'n': 4
}

blockMesh = ParsedBlockMeshDict(args).run()
```

### Parallel Boundary Conditions

```python
# Boundary conditions work automatically with parallel decomposition
# No special configuration needed
```

## Performance Considerations

### Domain Decomposition

| Decomposition | Communication | Use Case |
|--------------|-------------|---------|
| scotch | Low | Uniform distributions |
| hierarchical | Medium | Spatial locality |
| manual | High | Custom control |

### Load Balancing

| Balancing | Overhead | Use Case |
|----------|----------|---------|
| Dynamic | Medium | Automatic redistribution |
| Static | Low | No redistribution |
| None | Low | May cause imbalance |

### Solver Parallelization

| Solver | Parallel Support | Performance |
|--------|------------------|---------|
| simpleFoam | Yes | Good speedup |
| pimpleFoam | Yes | Better convergence |
| PISO | Yes | Pressure-implicit parallel |
| interFoam | Yes | Transient parallel |

## Common Issues and Solutions

### Load Imbalance

**Problem**: Uneven processor load

**Solutions**:
- Use dynamic load balancing
- Try different decomposition method
- Increase number of subdomains
- Check processor performance

### Communication Overhead

**Problem**: Excessive communication time

**Solutions**:
- Reduce frequency of boundary exchanges
- Use larger subdomains
- Batch communication operations
- Check network configuration

### Memory Issues

**Problem**: Out of memory

**Solutions**:
- Reduce mesh resolution
- Use fewer subdomains
- Reduce time step size
- Check memory usage per processor

### Synchronization Issues

**Problem**: Deadlock or race conditions

**Solutions**:
- Check for proper synchronization
- Avoid unnecessary barriers
- Use appropriate communication patterns
- Verify thread safety

## Best Practices

1. **Start with serial**: Verify correctness before parallelizing
2. **Choose appropriate decomposition**: Match problem characteristics
3. **Monitor performance**: Track load balance and communication
4. **Use appropriate solver**: Ensure solver supports parallelism
5. **Test scalability**: Verify strong and weak scaling
6. **Handle exceptions**: Catch and handle parallel errors
