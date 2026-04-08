# Transient Problems

## Time-Stepping Framework

For transient PDEs, march the solution forward in time:

```
u(t=0) = u₀  (initial condition)
For n = 1, 2, 3, ...:
    Solve: F(u^{n+1}, u^{n}, u^{n-1}, ...) = 0
    Save u^{n+1}
```

## Transient Diffusion

PDE: `∂u/∂t = ∇·(α∇u) + f`

### Backward Euler (First-Order)

```python
V = fem.functionspace(mesh, ("Lagrange", 1))
u = fem.Function(V)      # current solution
u_n = fem.Function(V)    # previous (initial at t=0)

v = TestFunction(V)

# Parameters
dt = 0.01
f = fem.Constant(mesh, 0.0)
alpha = fem.Constant(mesh, 1.0)

# Weak form at time t^{n+1}:
# (u^{n+1} - u^n)/dt = α ∇²u^{n+1} + f
F = (u - u_n) / dt * v * dx + alpha * dot(grad(u), grad(v)) * dx - f * v * dx

# Time stepping loop
t = 0.0
for n in range(num_steps):
    t += dt
    solve(F == 0, u, bcs)
    u_n.x.array[:] = u.x.array[:]  # Copy to previous
```

### Crank-Nicolson (Second-Order)

```python
# Average of explicit and implicit
# (u^{n+1} - u^n)/dt = (1/2)(α∇²u^{n+1} + α∇²u^n)
F = (u - u_n) / dt * v * dx \
    + 0.5 * alpha * dot(grad(u), grad(v)) * dx \
    + 0.5 * alpha * dot(grad(u_n), grad(v)) * dx \
    - f * v * dx
```

### BDF2 (Second-Order, More Stable)

```python
# Requires two previous values: u_n, u_nm1
u_nm1 = fem.Function(V)  # two steps back

# BDF2: (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = ...
F = (1.5 * u - 2.0 * u_n + 0.5 * u_nm1) / dt * v * dx \
    + alpha * dot(grad(u), grad(v)) * dx - f * v * dx
```

## Output in Transient

```python
# Save to file at intervals
from dolfinx import io

output_file = io.VTXWriter(mesh.comm, "u.bp", [u], engine="BP4")
output_file.write(t=0.0)

for n in range(num_steps):
    solve(F == 0, u, bcs)
    t += dt
    if n % output_interval == 0:
        output_file.write(t)
```

## State Update Discipline

At the end of each timestep, verify the solution was computed and update
the previous state:

```python
for n in range(num_steps):
    # 1. Solve
    solve(F == 0, u, bcs)

    # 2. Write output if needed
    if n % output_interval == 0:
        output_file.write(t)

    # 3. Update previous state
    u_n.x.array[:] = u.x.array[:]

    # 4. Advance time
    t += dt
```

**Critical:** If you forget to update `u_n`, the timestepper will use
stale data and the solution will be semantically broken.

## Adaptive Timestep

For problems with varying time scales:

```python
t = 0.0
dt = 0.01
dt_min = 1e-6
dt_max = 0.1

for n in range(num_steps):
    # Solve with current dt
    solve(F == 0, u, bcs)

    # Estimate local error (e.g., from two methods)
    error = compute_error(u, u_n)

    if error > tolerance:
        # Reject and reduce timestep
        dt *= 0.5
        continue
    else:
        # Accept and adjust
        u_n.x.array[:] = u.x.array[:]
        t += dt
        if error < 0.1 * tolerance:
            dt *= 1.2  # Grow timestep if very accurate
```

## Coupled Transient Systems

For multiple fields evolving together:

```python
# Stokes transient (velocity u, pressure p)
u = fem.Function(V)      # velocity
p = fem.Function(Q)      # pressure
u_n = fem.Function(V)
p_n = fem.Function(Q)

# Coupled system
F_u = (u - u_n) / dt * v * dx + viscosity * dot(grad(u), grad(v)) * dx \
      - p * div(v) * dx - dot(f, v) * dx
F_p = div(u) * q * dx

F = F_u + F_p

solve(F == 0, [u, p], bcs)
```

## Real-World Considerations

### Stabilization for Advection-Dominated

```python
# Add SUPG (Streamline Upwind Petrov-Galerkin) for advection
h = CellDiameter(mesh)
v = TestFunction(V)
Pe = h / (2 * abs(velocity)) * dot(velocity, grad(u))
tau = h / (2 * |velocity|) * (1 - exp(-Pe)) / Pe

F += tau * dot(velocity, grad(v)) * dot(velocity, grad(u))
```

### Conservation Properties

For conservative transport:

```python
# Ensure ∫u dx = ∫u₀ dx (mass conservation)
u_mean = assemble(u * dx) / assemble(1 * dx(mesh))
u.vector[:] *= initial_mean / u_mean
```

## Timestep Selection

| Physics | Stability Criterion | Suggested dt |
|---------|-------------------|--------------|
| Heat diffusion | `dt < dx²/(2α)` | Start conservative |
| Advection | `dt < dx/|u|` (CFL) | CFL < 1 |
| Wave propagation | `dt < dx/c` | CFL < 1 |
| Reaction-diffusion | Min of above | Combine criteria |
| Navier-Stokes (implicit) | `dt < dx²/ν` | For stability |

## Monitoring

Print progress during long runs:

```python
for n in range(num_steps):
    solve(F == 0, u, bcs)

    if mesh.comm.rank == 0 and n % 100 == 0:
        print(f"t = {t:.3f}, max(u) = {u.x.array.max():.4f}")
```
