#!/usr/bin/env python3
"""
Basic Poisson equation solver template for FEniCS.

Solves: -∇²u = f in Ω, u = 0 on ∂Ω

Usage:
    python scripts/template_poisson.py
"""

import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Parameters
L = 1.0              # Domain size
nx = 32               # Grid resolution

# Create mesh
'print("Creating mesh...")
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)

# Create function space (P1)
V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dfx.fem.Constant(mesh, 1.0)

# Variational form
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Boundary conditions
def boundary(x):
    return np.logical_or(
        np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
        np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)
    )

u0 = dfx.fem.Function(V)
u0.x.array[:] = 0.0
bc = dfx.fem.dirichletbc(u0, dfx.fem.locate_dofs_geometrical(V, boundary))

# Solve
print("Solving Poisson equation...")
u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, [bc], u_h)
dfx.nls.petsc.solve(problem)

# Output
print(f"Solution range: [{u_h.x.array.min():.6f}, {u_h.x.array.max():.6f}]")

# Save to file
with dfx.io.XDMFFile(MPI.COMM_WORLD, "poisson_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_h)

print("Solution saved to poisson_solution.xdmf")
