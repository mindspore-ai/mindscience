#!/usr/bin/env python3
"""
Stokes flow solver template for FEniCS.

Solves: -∇²u + ∇p = 0, ∇·u = 0 with Taylor-Hood elements

Usage:
    python scripts/template_stokes.py
"""

import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Parameters
nx = 32               # Grid resolution
nu = 1.0              # Viscosity

# Create mesh
print("Creating mesh...")
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)

# Taylor-Hood function space (P2-P1)
P2 = dfx.fem.functionspace(mesh, ("Lagrange", 2))
P1 = dfx.fem.functionspace(mesh, ("Lagrange", 1))
V = dfx.fem.functionspace(mesh, [P2, P1])

# Trial and test functions
(u, p) = ufl.TrialFunctions(V)
(v, q) = ufl.TestFunctions(V)

# Source term
f = dfx.fem.Constant.function(mesh, (0.0, 0.0))

# Variational form
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.div(v) * p + ufl.div(u) * q) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Boundary conditions
def walls(x):
    return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))

def lid(x):
    return np.isclose(x[1], 1.0)

# No-slip on walls
u0 = dfx.fem.Function(P2)
u0.x.array[:] = 0.0
bc_walls = dfx.fem.dirichletbc(u0, dfx.fem.locate_dofs_geometrical(P2, walls))

# Moving lid
u_lid = dfx.fem.Function(P2)
u_lid.x.array[:] = 1.0
bc_lid = dfx.fem.dirichletbc(u_lid, dfx.fem.locate_dofs_geometrical(P2, lid))

bcs = [bc_walls, bc_lid]

# Block preconditioner options
options = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_pc_type": "jacobi",
    "ksp_rtol": 1e-10
}

# Solve
print("Solving Stokes equations...")
u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, bcs, u_h, petsc_options=options)
dfx.nls.petsc.solve(problem)

# Extract velocity and pressure
u_sol, p_sol = u_h.sub(0), u_h.sub(1)

# Output
print(f"Velocity range: [{u_sol.x.array.min():.6f}, {u_sol.x.array.max():.6f}]")
print(f"Pressure range: [{p_sol.x.array.min():.6f}, {p_sol.x.array.max():.6f}]")

# Save to file
with dfx.io.XDMFFile(MPI.COMM_WORLD, "stokes_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_h)

print("Solution saved to stokes_solution.xdmf")
