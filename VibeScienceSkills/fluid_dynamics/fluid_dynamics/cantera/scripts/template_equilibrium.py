#!/usr/bin/env python3
"""
Chemical equilibrium calculation template for Cantera.

Usage:
    python scripts/template_equilibrium.py
"""

import cantera as ct
import numpy as np

# Create gas phase
gas = ct.Solution('gri30.yaml', 'gas')

# Set initial conditions
gas.TPX = 2000, 101325, 'H2:2, O2:1'

# Equilibrate at constant T, P
gas.equilibrate('HP', solver='gibbs')

# Output results
print(f"Equilibrium temperature: {gas.T} K")
print(f"Equilibrium pressure: {gas.P} Pa")
print(f"\nEquilibrium composition:")
for i, name in enumerate(gas.species_names):
    print(f"  {name}: {gas.X[i]:.6f}")

# Calculate equilibrium constants
K = gas.equilibrium_constants
print(f"\nEquilibrium constants:")
for i in range(gas.n_reactions):
    if gas.reaction(i).reversible:
        print(f"  Reaction {i}: {gas.reaction_equation(i)}")
        print(f"    K = {K[i]:.6e}")
