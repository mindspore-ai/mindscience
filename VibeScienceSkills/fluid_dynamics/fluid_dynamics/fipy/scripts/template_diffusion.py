#!/usr/bin/env python3
"""
Basic diffusion equation template for FiPy.

Solves: ∂φ/∂t = D∇²φ

Usage:
    python scripts/template_diffusion.py
"""

from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, MatplotlibViewer
import numpy as np

# Parameters
D = 1.0              # Diffusion coefficient
L = 1.0              # Domain size
N = 100              # Grid resolution (NxN)
dx = L / N            # Grid spacing
dt = 0.0001          # Time step
steps = 1000          # Number of time steps

# Create mesh
mesh = Grid2D(nx=N, ny=N, dx=dx, dy=dx)

# Create variable
phi = CellVariable(mesh=mesh, name='phi', value=0.)

# Initial condition: Gaussian pulse in center
X, Y = mesh.cellCenters
r2 = (X - L/2)**2 + (Y - L/2)**2
phi.setValue(np.exp(-r2 / 0.01))

# Boundary conditions (zero flux by default)
# Uncomment for fixed boundaries:
# phi.constrain(0., where=mesh.facesLeft)
# phi.constrain(0., where=mesh.facesRight)
# phi.constrain(0., where=mesh.facesTop)
# phi.constrain(0., where=mesh.facesBottom)

# Create equation
eq = TransientTerm() == DiffusionTerm(coeff=D)

# Create viewer
viewer = MatplotlibViewer(vars=phi, datamin=0., datamax=1.)

# Time stepping
print(f"Solving diffusion equation for {steps} steps...")
for step in range(steps):
    eq.solve(var=phi, dt=dt)
    
    # Update visualization every 10 steps
    if step % 10 == 0:
        viewer.plot()
        print(f"Step {step}/{steps}")

print("Simulation complete!")
print(f"Final phi range: [{phi.min()}, {phi.max()}]")
