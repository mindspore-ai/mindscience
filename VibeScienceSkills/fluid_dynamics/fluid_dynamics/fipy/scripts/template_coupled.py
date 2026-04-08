#!/usr/bin/env python3
"""
Coupled equations template for FiPy.

Solves two coupled diffusion equations:
∂φ/∂t = D₁∇²φ + αψ
∂ψ/∂t = D₂∇²ψ + βφ

Usage:
    python scripts/template_coupled.py
"""

from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm, MatplotlibViewer
import numpy as np

# Parameters
D1 = 1.0              # Diffusion coefficient for φ
D2 = 0.5              # Diffusion coefficient for ψ
alpha = 0.1            # Coupling coefficient (effect of ψ on φ)
beta = 0.2             # Coupling coefficient (effect of φ on ψ)
L_domain = 1.0         # Domain size
N = 100                # Grid resolution
dx = L_domain / N      # Grid spacing
dt = 0.0001            # Time step
steps = 1000           # Number of time steps

# Create mesh
mesh = Grid2D(nx=N, ny=N, dx=dx, dy=dx)

# Create variables
phi = CellVariable(mesh=mesh, name='phi', value=0.)
psi = CellVariable(mesh=mesh, name='psi', value=0.)

# Initial conditions
X, Y = mesh.cellCenters
phi.setValue(np.exp(-((X - 0.3)**2 + (Y - 0.5)**2) / 0.01))
psi.setValue(np.exp(-((X - 0.7)**2 + (Y - 0.5)**2) / 0.01))

# Boundary conditions (zero flux by default)
# Uncomment for fixed boundaries:
# phi.constrain(0., where=mesh.facesLeft)
# psi.constrain(0., where=mesh.facesRight)

# Create coupled equations
# Equation 1: ∂φ/∂t = D₁∇²φ + αψ
eq1 = (TransientTerm(var=phi)
       == DiffusionTerm(coeff=D1, var=phi)
       + ImplicitSourceTerm(alpha * psi, var=phi))

# Equation 2: ∂ψ/∂t = D₂∇²ψ + βφ
eq2 = (TransientTerm(var=psi)
       == DiffusionTerm(coeff=D2, var=psi)
       + ImplicitSourceTerm(beta * phi, var=psi))

# Couple equations
coupled_eq = eq1 & eq2

# Create viewers
viewer_phi = MatplotlibViewer(vars=phi, datamin=0., datamax=1.)
viewer_psi = MatplotlibViewer(vars=psi, datamin=0., datamax=1.)

# Time stepping
print(f"Solving coupled equations for {steps} steps...")
print(f"Coupling: alpha={alpha}, beta={beta}")

for step in range(steps):
    coupled_eq.solve(dt=dt)
    
    # Update visualization every 10 steps
    if step % 10 == 0:
        viewer_phi.plot()
        viewer_psi.plot()
        print(f"Step {step}/{steps}, φ: [{phi.min():.3f}, {phi.max():.3f}], ψ: [{psi.min():.3f}, {psi.max():.3f}]")

print("Simulation complete!")
print(f"Final φ range: [{phi.min():.3f}, {phi.max():.3f}]")
print(f"Final ψ range: [{psi.min():.3f}, {psi.max():.3f}]")

# Compute coupling strength
coupling = (phi * psi).cellVolumeAverage * mesh.cellVolumes.sum()
print(f"Total coupling: {coupling:.6f}")
