# Error Recovery

## Common Errors and Solutions

### Solver Errors

#### SCF Not Converging (Frequency Domain)

**Symptoms:**
- Solver stops before reaching requested accuracy
- Warning: "Adaptive mesh refinement did not converge"

**Causes:**
- Mesh too coarse
- Complex geometry
- High-Q structure

**Solutions:**
```
1. Increase initial mesh density
2. Enable adaptive mesh refinement
3. Increase maximum number of passes
4. Use lower frequency first, then interpolate
```

#### Time Step Too Small (Transient)

**Symptoms:**
- Simulation runs very slowly
- Warning: "Time step reduced for stability"

**Causes:**
- Small mesh cells
- High dielectric constant materials
- Courant limit exceeded

**Solutions:**
```
1. Increase minimum mesh cell size
2. Use sub-gridding for small features
3. Enable time step limiting
4. Consider Frequency Domain solver
```

#### Memory Overflow

**Symptoms:**
- Error: "Out of memory"
- Simulation crashes

**Causes:**
- Too many mesh cells
- Insufficient RAM
- Memory leak

**Solutions:**
```
1. Reduce mesh density
2. Use symmetry boundaries
3. Increase available memory
4. Use 64-bit version
5. Enable memory-saving options
```

### Mesh Errors

#### Mesh Generation Failed

**Symptoms:**
- Error: "Mesh generation failed"
- No mesh created

**Causes:**
- Geometry errors
- Overlapping faces
- Too small features

**Solutions:**
```
1. Check geometry for errors
2. Repair overlapping faces
3. Increase minimum feature size
4. Use local mesh refinement
```

#### Poor Mesh Quality

**Symptoms:**
- Warning: "Poor mesh quality detected"
- Inaccurate results

**Causes:**
- High aspect ratio cells
- Skewed elements
- Insufficient refinement

**Solutions:**
```
1. Enable mesh quality optimization
2. Use curvature refinement
3. Increase mesh density
4. Check geometry for small features
```

### Port Errors

#### Port Not Defined

**Symptoms:**
- Error: "No port defined"
- Simulation cannot start

**Causes:**
- Missing port definition
- Incorrect port type

**Solutions:**
```
1. Add waveguide or discrete port
2. Check port face selection
3. Verify port impedance
```

#### Port Reflection Too High

**Symptoms:**
- S11 > 0 dB at port
- Unphysical results

**Causes:**
- Port too close to discontinuity
- Incorrect port size
- Wrong mode selected

**Solutions:**
```
1. Move port further from discontinuity
2. Adjust port size to cover mode
3. Check mode pattern
4. Use de-embedding
```

### Boundary Errors

#### Open Boundary Reflection

**Symptoms:**
- Spurious reflections
- Incorrect far-field results

**Causes:**
- PML too close to structure
- Insufficient padding

**Solutions:**
```
1. Increase open boundary distance (λ/4 minimum)
2. Increase PML layers
3. Use open add space
```

#### Symmetry Error

**Symptoms:**
- Incorrect field pattern
- Wrong S-parameters

**Causes:**
- Wrong symmetry type
- Structure not symmetric

**Solutions:**
```
1. Verify structure symmetry
2. Check symmetry plane orientation
3. Use correct boundary type (E or H)
```

### License Errors

#### License Not Available

**Symptoms:**
- Error: "License checkout failed"
- Cannot start simulation

**Causes:**
- No license available
- License server down
- Wrong license type

**Solutions:**
```
1. Check license server status
2. Verify license file
3. Contact administrator
4. Use different solver if available
```

### Cluster Errors

#### Job Timeout

**Symptoms:**
- Job killed by scheduler
- Incomplete results

**Causes:**
- Walltime exceeded
- Insufficient time requested

**Solutions:**
```
1. Request longer walltime
2. Enable checkpoint
3. Restart from checkpoint
4. Optimize mesh for faster simulation
```

#### MPI Communication Error

**Symptoms:**
- Error: "MPI communication failed"
- Job crashes

**Causes:**
- Network issues
- Node failure
- Memory overflow on one node

**Solutions:**
```
1. Check network connectivity
2. Reduce processes per node
3. Increase memory per node
4. Use checkpoint for recovery
```

## Debugging Checklist

1. **Check geometry** for errors
2. **Verify mesh** quality
3. **Validate ports** and excitations
4. **Confirm boundaries** are correct
5. **Review solver** settings
6. **Check resources** (memory, time)
7. **Examine log files** for warnings

## Recovery Strategies

| Strategy | When to Use |
|----------|-------------|
| Restart from checkpoint | Long simulation crashed |
| Reduce mesh density | Memory overflow |
| Use symmetry | Domain too large |
| Change solver | Convergence issues |
| Simplify geometry | Mesh errors |
