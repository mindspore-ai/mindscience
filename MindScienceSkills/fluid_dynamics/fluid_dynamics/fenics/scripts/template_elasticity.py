#!/usr/bin/env python3
"""
Linear elasticity solver template for FEniCS.

Solves: ∇·σ = f, σ = λ tr(ε) I + 2μ ε, ε = (∇u + ∇uᵀ)/2

Usage:
    python scripts/template_elasticity.py
"""

import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Parameters
nx = 32               # Grid resolution
E = 1.0e5              # Young's modulus
nu_mat = 0.3           # Poisson's ratio

# Create mesh
print("Creating mesh...")
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)

# Vector function space (P1)
V = dfx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))

# Trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Lame parameters
mu = E / (2 * (1 + nu_mat))
lmbda = E * nu_mat / ((1 + nu_mat) * (1 - 2 * nu_mat))

# Strain and stress
def epsilon(u):
    return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

def sigma(u):
    return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

# Source term (body force - gravity)
f = dfx.fem.Constant.function(mesh, (0.0, -9.81))

# Variational form
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Boundary conditions
def clamped_boundary(x):
    return np.isclose(x[0], 0.0)

u0 = dfx.fem.Function(V)
u0.x.array[:] = [0.0, 0.0]
bc = dfx.fem.dirichletbc(u0, dfx.fem.locate_dofs_geometrical(V, clamped_boundary))

# Solve
print("Solving elasticity problem...")
u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, [bc], u_h)
dfx.nls.petsc.solve(problem)

# Output
print(f"Displacement range: [{u_h.x.array.min():.6f}, {u_h.x.array.max():.6f}]")

# Compute strain energy
eps = epsilon(u_h)
strain_energy = dfx.fem.assemble_scalar(0.5 * ufl.inner(sigma(u_h), eps) * ufl.dx)
print(f"Strain energy: {strain_energy:.6f}")

# Save to file
with dfx.io.XDMFFile(MPI.COMM_WORLD, "elasticity_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_h)

print("Solution saved to elasticity_solution.xdmf")
