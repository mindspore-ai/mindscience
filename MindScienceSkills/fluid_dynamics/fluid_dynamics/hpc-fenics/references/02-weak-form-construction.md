# Weak Form Construction (UFL)

## The Weak Form Philosophy

The finite element method solves PDEs in **weak form**:

1. Multiply the PDE by a test function `v`
2. Integrate over the domain Ω
3. Apply integration by parts (Green's identity)
4. Impose boundary conditions

This turns differential equations into a linear algebraic system `Au = b`.

## UFL (Unified Form Language)

UFL is the domain-specific language in FEniCS for describing finite element
forms. It is shared (with API differences) between classic FEniCS and DOLFINx.

## Basic Patterns

### Linear Poisson Equation

PDE: `-∇²u = f` in Ω, `u = 0` on ∂Ω

Weak form: `∫(∇u · ∇v) dx = ∫(f · v) dx`

```python
# DOLFINx
u = fem.Function(V)          # unknown (trial)
v = fem.FunctionSpace(mesh, ("Lagrange", 1))
u = TrialFunction(V)        # for linear problem
v = TestFunction(V)         # test function

a = dot(grad(u), grad(v)) * dx  # bilinear form
L = f * v * dx                   # linear form
```

### Breaking Down UFL Operators

| UFL | Meaning |
|-----|---------|
| `grad(u)` | ∇u (gradient) |
| `div(u)` | ∇·u (divergence) |
| `dot(u, v)` | u · v (dot product) |
| `inner(u, v)` | u : v (inner product, same as dot for vectors) |
| `dx` | Integration over domain |
| `ds` | Integration over exterior boundary |
| `dS` | Integration over interior facets |

### Explicit vs Implicit Boundary Conditions

In FEniCS, the weak form does NOT include boundary conditions — they are
applied separately as `DirichletBC` or `fem.dirichletbc`. This is the
**penalty/elimination approach**.

## Common PDEs and Their Weak Forms

### Diffusion Equation

PDE: `∂u/∂t = ∇·(α∇u) + f`

Semidiscrete weak form:

```python
# For linear elements
u = TrialFunction(V)
v = TestFunction(V)

a = u * v * dx + dt * alpha * dot(grad(u), grad(v)) * dx
L = u_n * v * dx + dt * f * v * dx
```

### Linear Elasticity

PDE: `-∇·σ(u) = f` with σ(u) = λ tr(ε)I + 2μ ε

Strain: `ε(u) = (∇u + ∇uᵀ)/2`

```python
# 2D elasticity
def epsilon(u):
    return 0.5 * (grad(u) + grad(u).T)

def sigma(u, mu, lam):
    return 2.0 * mu * epsilon(u) + lam * div(u) * Identity(2)

u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u, mu, lam), epsilon(v)) * dx
L = dot(f, v) * dx
```

### Stokes Flow (Mixed Form)

PDE:
- `-∇²u + ∇p = 0`
- `∇·u = 0`

```python
# Taylor-Hood elements: P2 for velocity, P1 for pressure
V = VectorFunctionSpace(mesh, ("Lagrange", 2))
Q = FunctionSpace(mesh, ("Lagrange", 1))

u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

a = dot(grad(u), grad(v)) * dx - div(v) * p * dx - q * div(u) * dx
L = dot(f, v) * dx
```

## Nonlinear Weak Forms

For nonlinear PDEs, define the **residual** F(u; v) = 0:

```python
# Nonlinear: -∇·(k(u)∇u) = f  (nonlinear diffusion)
u = Function(V)        # unknown, NOT TrialFunction
v = TestFunction(V)

k_u = k(u)  # coefficient depends on u
F = dot(k_u * grad(u), grad(v)) * dx - f * v * dx
```

Then solve `F == 0` for `u`.

## Form Verification

Always check:
1. **Rank of the unknown** — scalar → scalar space, vector → vector space
2. **Domain dimension** — matches mesh dimension
3. **Boundary terms** — `ds` vs `dx` correct for the physics

### Common UFL Mistakes

```python
# WRONG: using dot on scalar arguments
dot(u, v)  # u and v are scalars — use u * v

# WRONG: shape mismatch
grad(u) + u  # grad(u) is vector, u is scalar — use div(grad(u)) + u

# CORRECT: scalar dot
u * v  # for scalars

# CORRECT: vector dot
dot(grad(u), grad(v))  # for scalar PDEs
```

## Integration Measures

| Measure | Meaning |
|---------|---------|
| `dx` | Volume integral over domain |
| `ds` | Exterior surface integral |
| `dS` | Interior facet integral (for DG) |
| `dC` | Exterior facet integral |

For mixed-dimensional coupling:

```python
# Interface integral: dot(jump(u), n) * dS
from ufl import jump
```
