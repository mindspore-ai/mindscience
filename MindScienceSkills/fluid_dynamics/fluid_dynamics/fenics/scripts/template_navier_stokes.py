#!/usr/bin/env python3
"""
Transient Navier-Stokes solver template for FEniCS.

Solves: ∂u/∂t + (u·∇)u - ν∇²u + ∇p = 0, ∇·u = 0

Usage:
    python scripts/template_navier_stokes.py
"""

import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Parameters
nx = 32               # Grid resolution
nu = 0.01              # Kinematic viscosity
T = 1.0                # Final time
dt = 0.01               # Time step
num_steps = int(T / dt)

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

# Previous time step
u_n = dfx.fem.Function(P2)

# Source term
f = dfx.fem.Constant.function(mesh, (0.0, 0.0))

# Variational form (BDF1)
a = (ufl.inner(u, v) * ufl.dx
      + dt * ufl.inner(ufl.grad(u), ufl.grad(u_n)) * v * ufl.dx
      + dt * nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
      - dt * ufl.div(v) * p * ufl.dx
      + dt * ufl.div(u) * q * ufl.dx)
L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f, v) * ufl.dx

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
print("Solving Navier-Stokes equations...")
u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, bcs, u_h, petsc_options=options)

for step in range(num_steps):
    t = step * dt
    
    # Solve
    dfx.nls.petsc.solve(problem)
    
    # Update previous time step
    u_n.x.array[:] = u_h.sub(0).x.array[:]
    
    if step % 10 == 0:
        u_sol, p_sol = u_h.sub(0), u_h.sub(1)
        print(f"Step {step}/{num_steps}, t = {t:.3f}")
        print(f"  Velocity range: [{u_sol.x.array.min():.6f}, {u_sol.x.array.max():.6f}]")
        print(f"  Pressure range: [{p_sol.x.array.min():.6f}, {p_sol.x.array.max():.6f}]")

# Save final solution
with dfx.io.XDMFFile(MPI.COMM_WORLD, "navier_stokes_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_h)

print("Solution saved to navier_stokes_solution.xdmf")
