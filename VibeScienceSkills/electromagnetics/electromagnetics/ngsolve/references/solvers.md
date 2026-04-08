# Solvers in NGsolve

NGsolve provides various solver types for different problem types.

## Solver Types

### Poisson Solver

For electrostatic problems:

```python
import ngsolve as ns

# Create mesh
mesh = ns.Mesh()
mesh.add_rect(0, 0, 1, 1)

# Create function space
V = ns.FunctionSpace('V')
V.set_parameter_order('u', 'u')

# Define material
u = ns.Constant('u')
u.set_value(1.0)
V.set_material(u)

# Define problem
problem = ns.Poisson('u', mesh)

# Create solver
solver = ns.PoissonSolver('u', mesh, problem)
solver.solve()
```

### Helmholtz Solver

For time-harmonic electromagnetic problems:

```python
import ngsolve as ns

# Create mesh
mesh = ns.Mesh()
mesh.add_rect(0, 0, 1, 1)

# Function space
V = ns.FunctionSpace('V')
V.set_parameter_order('u', 'u', 'E', 'H')

# Material
epsilon = ns.Constant('epsilon_r')
epsilon.set_value(4.0)
V.set_material(epsilon)

# Source
f = ns.Constant('f')
f.set_value(1.0)
V.set_source(f)

# Problem
problem = ns.HelmholtzEquation('V', mesh, 
                                        degree=ns.grad(V)**2,
                                        frequency=2*np.pi)

# Solver
solver = ns.HelmholtzSolver('V', mesh, problem)
solver.solve()
```

### Wave Solver

For time-domain wave equations:

```python
import ngsolve as ns

# Create mesh
mesh = ns.Mesh()
mesh.add_rect(0, 0, 1, 1)

# Function space
V = ns.FunctionSpace('V')
V.set_parameter_order('u', 'u')

# Material
epsilon = ns.Constant('epsilon_r')
epsilon.set_value(4.0)
V.set_material(epsilon)

# Source
f = ns.Constant('f')
f.set_value(1.0)
V.set_source(f)

# Problem
problem = ns.WaveEquation('V', mesh, 
                              degree=ns.grad(V)**2,
                              frequency=2*np.pi)

# Solver
solver = ns.WaveSolver('V', mesh, problem)
solver.solve()
```

### Navier-Stokes Solver

For incompressible fluid flow:

```python
import ngsolve as ns

# Create mesh
mesh = ns.Mesh()
mesh.add_rect(0, 0, 1, 1)

# Function space
V = ns.FunctionSpace('V', 'u')
V.set_parameter_order('u', 'u')

# Material
mu = ns.Constant('mu')
mu.set_value(1.0)
V.set_material(mu)

# Problem
problem = ns.NavierStokes('V', mesh, 
                             degree=ns.grad(V)**2)

# Solver
solver = ns.NavierStokesSolver('V', mesh, problem)
solver.solve()
```

## Solver Configuration

### Solver Parameters

```python
# Create solver with parameters
solver = ns.PoissonSolver('u', mesh, problem,
                                   linear_solver='direct',
                                   preconditioner='ilu',
                                   absolute_tolerance=1e-6,
                                   relative_tolerance=1e-6,
                                   max_iterations=1000)
```

### Solver Execution

```python
# Solve problem
solver.solve()

# Solve with monitoring
class MyMonitor(ns.ProgressMonitor):
    def __init__(self):
        self.step = 0
    
    def __call__(self, mesh):
        self.step += 1
        if self.step % 10 == 0:
            print(f"Step {self.step}")

monitor = MyMonitor()
solver.solve(monitor=monitor)
```

### Solver Output

```python
# Get solution
u = V.get_subfunction('u')
u_array = u.vector().get_array()

# Get solver statistics
stats = solver.get_solver_stats()
print(f"Iterations: {stats['iterations']}")
print(f"Residual: {stats['residual']}")
```

## Solver Selection

### Problem Type Guidelines

1. **Electrostatics**: Use Poisson solver
2. **Time-harmonic**: Use Helmholtz solver
3. **Wave equations**: Use Wave solver
4. **Fluid flow**: Use Navier-Stokes solver

### Solver Performance

1. **Small problems**: Use direct solver
2. **Large problems**: Use iterative solver
3. **Parallel computing**: Enable MPI support
4. **Memory constraints**: Use iterative solver

## Best Practices

### Solver Configuration

1. **Choose appropriate solver**: Match solver to problem type
2. **Use preconditioners**: Improve convergence
3. **Set tolerances**: Balance accuracy and speed
4. **Monitor convergence**: Track solver progress

### Convergence

1. **Check residuals**: Verify solution convergence
2. **Monitor iterations**: Track iteration count
3. **Adjust tolerances**: Refine tolerances if needed
4. **Verify solution**: Check physical reasonableness

### Performance

1. **Use direct solver**: For small problems
2. **Use iterative solver**: For large problems
3. **Enable parallel computing**: Use MPI for distributed memory
4. **Profile solver**: Profile before optimizing

## Troubleshooting

### Convergence Issues

**Problem**: Solver doesn't converge

**Solutions**:
1. Increase mesh resolution
2. Improve preconditioner
3. Adjust tolerances
4. Check problem formulation

### Memory Issues

**Problem**: Out of memory errors

**solutions**:
1. Reduce mesh resolution
2. Use iterative solver
3. Reduce output frequency
4. Use parallel computing

### Performance Issues

**Problem**: Simulation is too slow

**solutions**:
1. Use appropriate solver type
2. Enable parallel computing
3. Profile solver
4. Optimize mesh quality