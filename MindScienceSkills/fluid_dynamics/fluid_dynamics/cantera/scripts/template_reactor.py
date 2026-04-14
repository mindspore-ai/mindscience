#!/usr/bin/env python3
"""
Simple batch reactor simulation template for Cantera.

Usage:
    python scripts/template_reactor.py
"""

import cantera as ct

# Create gas phase
gas = ct.Solution('gri30.yaml', 'gas')

# Set initial conditions
gas.TPX = 1500, 101325, 'H2:2, O2:1'

# Create constant pressure reactor
r = ct.IdealGasConstPressureReactor(gas)

# Create reactor network
sim = ct.ReactorNet([r])

# Set initial time
sim.set_initial_time(0.0)

# Advance in time
sim.advance(1e-3)

# Output results
print(f"Time: {sim.time} s")
print(f"Temperature: {gas.T} K")
print(f"Pressure: {gas.P} Pa")
print(f"Density: {gas.density} kg/m³")
print(f"\nSpecies:")
for i, name in enumerate(gas.species_names):
    print(f"  {name}: {gas.X[i]:.6f}")
