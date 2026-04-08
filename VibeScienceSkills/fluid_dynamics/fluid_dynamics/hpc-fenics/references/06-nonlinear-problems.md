# Nonlinear Problems

## Nonlinear PDE Pattern

For nonlinear PDEs, the weak form is:

```
Find u ∈ V such that F(u; v) = 0  for all v ∈ V̂
```

Where F is the **nonlinear residual** and u is a `Function` (not `TrialFunction`).

## DOLFINx NonlinearProblem

```python
from dolfinx import fem
from dolfinx.nls import NewtonSolver
from petsc4py import PETSc

# Define the residual F(u, v) = 0
u = fem.Function(V)  # The unknown
v = TestFunction(V)

F = dot(grad(u), grad(v)) * dx - u**3 * v * dx

# Define the Jacobian J = dF/du
J = derivative(F, u, TrialFunction(V))

# Create nonlinear problem
problem = fem.petsc.NonlinearProblem(F, u, bcs=bcs, J=J)

# Solve with Newton
solver = NewtonSolver(mesh.comm, problem)
solver.convergence_criterion = "residual"
solver.atol = 1e-8
solver.rtol = 1e-6
solver.max_it = 50
solver.report = True

u = solver.solve()
```

## Classic FEniCS Nonlinear Solve

```python
u = Function(V)

F = dot(grad(u), grad(v)) * dx - u**3 * v * dx

solve(F == 0, u, bcs)
```

FEniCS automatically computes the Jacobian for this form.

## Common Nonlinear PDEs

### p-Laplacian

PDE: `-∇·(|∇u|^{p-2} ∇u) = f`

```python
u = Function(V)
v = TestFunction(V)
p = Constant(mesh, 2.5)  # p > 2

F = dot(pow(dot(grad(u), grad(u)), (p-2)/2) * grad(u), grad(v)) * dx - f * v * dx
```

### Nonlinear Advection-Diffusion

PDE: `-∇·(D(u)∇u) + u·∇u = f`

```python
u = Function(V)
v = TestFunction(V)

F = dot(D(u) * grad(u), grad(v)) * dx + dot(u, grad(u)) * v * dx - f * v * dx
```

## Manual Newton Iteration

For more control, implement Newton manually:

```python
u = Function(V)  # Initial guess
v = TestFunction(V)
F = residual(u, v)  # Define residual function

tol = 1e-8
max_it = 50
for i in range(max_it):
    J = derivative(F, u)  # Jacobian
    solve(J == -F, du)    # Increment

    u.vector.axpy(-1.0, du.vector)  # u = u - du

    if norm(du.vector) < tol:
        print(f"Converged in {i+1} iterations")
        break
```

## Line Search

When Newton diverges, use line search:

```python
solver = NewtonSolver(mesh.comm, problem)
solver.line_search = "lagrange"  # or "cp" (critical point)
```

For the PETSc SNES backend:

```python
problem = fem.petsc.NonlinearProblem(F, u, bcs=bcs, J=J,
    petsc_options={"snes_type": "newtonls",
                   "snes_linesearch_type": "bt"})  # backtracking
```

## Initial Guess

A good initial guess is critical for nonlinear problems:

```python
# Start with linear solution
u_linear = Function(V)
solve(a_linear == L_linear, u_linear, bcs)

# Use as initial guess for nonlinear
u = u_linear.copy()
```

If no good guess is available:
- Use continuation (start with small parameter, increase)
- Use a coarse mesh solution as initial guess

## Convergence Criteria

| Criterion | Meaning | Use When |
|-----------|---------|----------|
| `"residual"` | `||F(u_k)|| < atol + rtol * ||F(u_0)||` | Default |
| `"incremental"` | `||u_k - u_{k-1}|| < atol + rtol * ||u_0||` | When residual is expensive |
| Both | Both must be satisfied | Tight tolerance needed |

## Convergence Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Diverges immediately | Bad initial guess | Start closer to solution |
| Diverges after several steps | Step size too large | Enable line search |
| Oscillates | Nonlinearity too strong | Use continuation |
| Converges very slowly | Preconditioner bad | Improve linear solver |
| "Divided by zero" | Jacobian singularity | Check BCs, initial guess |

## Continuation (Parameter Sweep)

For highly nonlinear problems:

```python
# Gradually increase the parameter
for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
    update_load(alpha)
    solve(F == 0, u, bcs, ...)
    save_results(u, alpha)
```

Use the previous solution as the next initial guess.

## PETSc SNES Options

For difficult nonlinear problems, use PETSc SNES directly:

```python
from petsc4py import PETSc

# Create SNES (Scalable Nonlinear Equation Solver)
snes = PETSc.SNES().create(mesh.comm)

# Set the residual and Jacobian
snes.setFunction(residual_petsc, F_vec)
snes.setJacobian(jacobian_petsc, J_mat)

# Set options
snes.setFromOptions()
snes.solve(None, u_vec)
```

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Forgetting `derivative()` | No Jacobian, slow convergence | Always provide J |
| Wrong BC for nonlinear | Solution can't satisfy BC | Use consistent BCs |
| Not updating coefficients | Wrong residual at each step | Use `u` directly in forms |
| Material nonlinearity ignored | Form doesn't depend on u | Check coefficient dependence |
