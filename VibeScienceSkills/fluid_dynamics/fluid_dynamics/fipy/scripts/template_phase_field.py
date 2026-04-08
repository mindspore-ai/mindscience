#!/usr/bin/env python3
"""
Allen-Cahn phase field model template.

Solves: ∂φ/∂t = L[ε²∇²φ - f'(φ)]
where f'(φ) = A * φ * (1 - φ) * (1 - 2φ)

Usage:
    python scripts/template_phase_field.py
"""

from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm, MatplotlibViewer
import numpy as np

# Parameters
L = 1.0              # Mobility
epsilon = 0.01        # Interface width
A = 1.0               # Energy barrier height
L_domain = 1.0         # Domain size
N = 100                # Grid resolution
dx = L_domain / N      # Grid spacing
dt = 0.0001            # Time step
steps = 1000           # Number of time steps

# Create mesh
mesh = Grid2D(nx=N, ny=N, dx=dx, dy=dx)

# Create order parameter
phi = CellVariable(mesh=mesh, name='order parameter', value=0.)

# Initial condition: random perturbation around 0.5
np.random.seed(42)
phi.setValue(0.5 + 0.1 * (np.random.random(mesh.numberOfCells) - 0.5))

# Boundary conditions (periodic by default)
# Uncomment for no-flux boundaries:
# phi.faceGrad.constrain(0., where=mesh.exteriorFaces)

# Create Allen-Cahn equation
# ∂φ/∂t = L[ε²∇²φ - A*φ*(1-φ)*(1-2φ)]
eq = (TransientTerm()
      == DiffusionTerm(coeff=L * epsilon**2)
      - ImplicitSourceTerm(L * A * phi * (1 - phi) * (1 - 2 * phi)))

# Create viewer
viewer = MatplotlibViewer(vars=phi, datamin=0., datamax=1.)

# Time stepping
print(f"Solving Allen-Cahn equation for {steps} steps...")
print(f"Interface width: {epsilon}, Grid spacing: {dx}")
print(f"Interface resolution: {epsilon/dx:.1f} grid points")

for step in range(steps):
    eq.solve(var=phi, dt=dt)
    
    # Update visualization every 10 steps
    if step % 10 == 0:
        viewer.plot()
        print(f"Step {step}/{steps}, φ range: [{phi.min():.3f}, {phi.max():.3f}]")

print("Simulation complete!")
print(f"Final φ range: [{phi.min():.3f}, {phi.max():.3f}]")

# Compute interface fraction (cells with 0.4 < φ < 0.6)
interface_cells = ((phi > 0.4) & (phi < 0.6)).sum()
print(f"Interface cells: {interface_cells} / {mesh.numberOfCells}")
