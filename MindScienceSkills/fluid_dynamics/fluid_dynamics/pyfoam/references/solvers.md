# Solvers

OpenFOAM provides multiple solvers for different types of fluid flow problems.

## Standard Solvers

### SIMPLE Solver

**Description**: Steady-state solver for incompressible flows

**Characteristics**:
- Semi-implicit method
- SIMPLE algorithm (Semi-Implicit Method for Pressure-Linked Equations)
- Fast convergence
- Good for steady-state flows

**Configuration**:
```python
applicationDict = {
    'solver': 'simpleFoam',
    'maxCo': 0.1,
    'maxAlphaCo': 0.1
}
```

**Use case**: Standard incompressible flows, steady-state solutions

### PIMPLE Solver

**Description**: Transient solver for incompressible flows

**Characteristics**:
- PISO (Pressure-Implicit with Splitting of Operators)
- Transient accuracy
- Good for unsteady flows
- Supports multiphase flows

**Configuration**:
```python
applicationDict = {
    'solver': 'pimpleFoam',
    'maxCo': 0.7,
    'maxAlphaCo': 0.3
}
```

**Use case**: Transient incompressible flows, multiphase flows

### PISO Solver

**Description**: Pressure-Implicit with Splitting of Operators solver

**Characteristics**:
- Pressure-implicit with momentum coupling
- PISO algorithm
- Good for incompressible flows
- Supports compressibility

**Configuration**:
```python
applicationDict = {
    'solver': 'PISO',
    'maxCo': 0.8,
    'maxAlphaCo': 0.2
}
```

**Use case**: Incompressible flows, multiphase flows with compressibility

## Compressible Solvers

### buoyantBoussinesqSimpleFoam

**Description**: Incompressible buoyancy-driven flow solver

**Characteristics**:
- Boussinesq approximation
- Incompressible flow
- Buoyancy force
- Good for natural convection

**Configuration**:
```python
applicationDict = {
    'solver': 'buoyantBoussinesqSimpleFoam'
}
```

### interFoam

**Description**: Incompressible flow solver

**Characteristics**:
- InterFoam formulation
- PISO algorithm
- Good for incompressible flows
- Transient simulation

**Configuration**:
```python
applicationDict = {
    'solver': 'interFoam'
}
```

## Multiphase Solvers

### reactingFoam

**Description**: Reacting multiphase flow solver

**Characteristics**:
- Chemical reactions
- Heat transfer
- Multiphase transport
- Species transport

**Configuration**:
```python
applicationDict = {
    'solver': 'reactingFoam',
    'chemistry': True
}
```

### compressibleInterFoam

**Description**: Compressible multiphase flow solver

**Characteristics**:
- Compressible multiphase flows
- Volume of fluid method
- Phase transport
- Species transport

**Configuration**:
```python
applicationDict = {
    'solver': 'compressibleInterFoam'
}
```

## Solver Selection Guide

| Flow Type | Recommended Solver | Reason |
|-----------|-------------------|---------|
| Steady incompressible | SIMPLE | Fast, stable |
| Transient incompressible | PIMPLE | Transient accuracy |
| Incompressible with buoyancy | buoyantBoussinesqSimpleFoam | Buoyancy effects |
| Incompressible with heat | interFoam | Heat transfer |
| Multiphase reacting | reactingFoam | Chemical reactions |
| Compressible multiphase | compressibleInterFoam | Volume of fluid |

## Solver Configuration

### Convergence Controls

```python
applicationDict = {
    'solver': 'simpleFoam',
    'maxCo': 0.1,        # Max Courant number
    'maxAlphaCo': 0.1,    # Max under-relaxation
    'nAlphaSubSteps': 2,      # Number of sub-cycles
    'nOuterCorrectors': 2,    # Number of outer correctors
}
```

### Time Stepping

```python
applicationDict = {
    'adjustableRunTime': True,   # Adaptive time stepping
    'maxCo': 0.3,             # Max Courant number
    'maxDeltaT': 1.0,          # Max temperature change
}
```

### Solver Tolerances

```python
applicationDict = {
    'solver': 'simpleFoam',
    'tolerance': 1e-6,        # Solver tolerance
    'relTol': 0.01,            # Relative tolerance
    'nNonOrthogonal': 1,      # Non-orthogonal correctors
}
```

## Solver Performance

### SIMPLE Solver Performance

**Typical performance**:
- Fast convergence (10-20 iterations)
- Low memory usage
- Good for 2D/3D problems
- Stable for steady-state flows

### PIMPLE Solver Performance

**Typical performance**:
- Slower convergence (20-50 iterations)
- Higher memory usage
- Better transient accuracy
- Good for unsteady flows

### PISO Solver Performance

**Typical performance**:
- Moderate convergence (30-100 iterations)
- Higher memory usage
- Best incompressible accuracy
- Good for multiphase flows

## Common Issues and Solutions

### Convergence Failures

**Problem**: Solver fails to converge

**Solutions**:
- Reduce time step size
- Improve mesh quality
- Check boundary conditions
- Adjust solver tolerances
- Try different solver

### Divergence

**Problem**: Solution diverges

**Solutions**:
- Check Courant number (reduce maxCo)
- Improve mesh quality
- Check boundary conditions
- Use appropriate differencing scheme
- Reduce time step size

### Slow Convergence

**Problem**: Convergence too slow

**Solutions**:
- Improve initial guess
- Use better differencing scheme
- Adjust solver tolerances
- Check mesh quality
- Try different solver

### Instability

**Problem**: Simulation becomes unstable

**Solutions**:
- Reduce time step size
- Improve mesh quality
- Check boundary conditions
- Use appropriate differencing scheme
- Try different solver

## Best Practices

1. **Start with SIMPLE**: Use SIMPLE solver for initial testing
2. **Monitor convergence**: Check residuals during simulation
3. **Choose appropriate solver**: Match solver to problem type
4. **Adjust tolerances**: Balance accuracy and speed
5. **Check mesh quality**: Poor mesh affects convergence
6. **Document settings**: Keep track of solver configurations
7. **Validate results**: Compare with benchmarks or analytical solutions
