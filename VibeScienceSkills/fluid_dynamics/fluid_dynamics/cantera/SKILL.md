---
name: cantera
description: Open-source chemical kinetics, thermodynamics, and transport library. Use when solving problems involving: (1) Chemical kinetics and reaction mechanisms, (2) Thermodynamic properties and phase equilibria, (3) Transport properties and multicomponent diffusion, (4) Combustion and flame modeling, (5) Electrochemical systems, (6) Fuel cells and batteries, (7) Plasma chemistry, (8) Thin film deposition, or (9) Multiphase chemical systems
license: BSD-3-Clause
metadata:
    skill-author: K-Dense Inc.
---

# Cantera

Open-source suite for chemical kinetics, thermodynamics, and transport processes.

## Overview

Cantera automates chemical kinetic, thermodynamic, and transport calculations for efficient incorporation of detailed chemical thermo-kinetics and transport models into simulations. It provides object-oriented phase models and generalized algorithms for exploring different phase models with minimal code changes.

## Quick Start

**Installation:**
```bash
conda install -c cantera cantera
# or
pip install cantera
```

**Basic thermodynamics:**
```python
import cantera as ct

# Create ideal gas phase
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'H2:1, O2:0.1'

# Compute properties
print(f"Temperature: {gas.T} K")
print(f"Pressure: {gas.P} Pa")
print(f"Density: {gas.density} kg/m³")
print(f"Enthalpy: {gas.enthalpy_mass} J/kg")
```

**Basic kinetics:**
```python
import cantera as ct

# Create gas phase with kinetics
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 1500, 101325, 'H2:2, O2:1'

# Create reactor
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

# Advance in time
sim.advance(1e-3)

print(f"Time: {sim.time} s")
print(f"Temperature: {gas.T} K")
print(f"Species: {gas.species_names}")
print(f"Mole fractions: {gas.X}")
```

## Core Workflow

### 1. Phase Input Files

Cantera uses YAML input files to define phases and species:

**Basic YAML structure:**
```yaml
phases:
  - name: gas
    thermo: ideal-gas
    kinetics: gas-reactions
    species: H2 O2 H2O OH H O2

species:
  - name: H2
    composition: {H: 2}
    thermo:
      NASA polyomial coefficients...

reactions:
  - equation: 2 H2 + O2 = 2 H2O
    rate: {A: 1.0e8, b: 0.0, c: 0.0}
```

**Common input files:**
- `gri30.yaml`: GRI-Mech 3.0 (30 species)
- `gri30_highT.yaml`: GRI-Mech 3.0 high temperature
- `h2o2.yaml`: H2/O2 mechanism
- `air.yaml`: Air composition

### 2. Creating Phases

**Ideal gas phase:**
```python
import cantera as ct

# From YAML file
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'N2:0.79, O2:0.21'

# From CTI file (legacy)
gas = ct.Solution('gas.cti', 'gas')
```

**Surface phase:**
```python
# Surface kinetics
surf = ct.Interface('surface.yaml', 'Pt_surf', [gas])
surf.coverages = {'H(s)': 0.5, 'O(s)': 0.3, 'Pt(s)': 0.2}
```

**Liquid phase:**
```python
# Liquid phase with Redlich-Kwong EOS
liquid = ct.Solution('liquid.yaml', 'liquid')
liquid.TPX = 300, 101325, 'H2O:1.0'
```

**Solid phase:**
```python
# Solid phase
solid = ct.Solution('solid.yaml', 'solid')
solid.TP = 300, 101325
```

### 3. Thermodynamic Properties

**State variables:**
```python
# Set state
gas.TPX = 300, 101325, 'H2:1, O2:0.1'  # T, P, composition
gas.TDX = 300, 1.0, 'H2:1, O2:0.1'    # T, density, composition
gas.TPX = 300, 101325, gas.X             # T, P, from mole fractions
gas.HP = gas.enthalpy_mass, 101325      # Enthalpy, pressure
```

**Thermodynamic properties:**
```python
# Molar properties
T = gas.T                    # Temperature (K)
P = gas.P                    # Pressure (Pa)
h = gas.enthalpy_mole          # Enthalpy (J/mol)
s = gas.entropy_mole           # Entropy (J/mol/K)
g = gas.gibbs_mole             # Gibbs energy (J/mol)
cp = gas.cp_mole               # Heat capacity (J/mol/K)
cv = gas.cv_mole               # Heat capacity (J/mol/K)
u = gas.int_energy_mole         # Internal energy (J/mol)

# Mass properties
h_mass = gas.enthalpy_mass       # Enthalpy (J/kg)
cp_mass = gas.cp_mass           # Heat capacity (J/kg/K)
density = gas.density            # Density (kg/m³)

# Partial molar properties
h_H2 = gas.partial_molar_enthalpies[gas.species_index('H2')]
g_H2 = gas.partial_molar_gibbs[gas.species_index('H2')]
```

**Species properties:**
```python
# Mole fractions
X = gas.X                    # Array of mole fractions
X_H2 = gas.X[gas.species_index('H2')]

# Mass fractions
Y = gas.Y                    # Array of mass fractions
Y_H2 = gas.Y[gas.species_index('H2')]

# Concentrations
C = gas.concentrations         # kmol/m³
C_H2 = gas.concentrations[gas.species_index('H2')]

# Chemical potentials
mu = gas.chemical_potentials     # J/mol
mu_H2 = mu[gas.species_index('H2')]
```

### 4. Kinetic Calculations

**Reaction rates:**
```python
# Forward and reverse rates
w_f = gas.forward_rates_of_progress   # kmol/m³/s
w_r = gas.reverse_rates_of_progress    # kmol/m³/s
w_net = w_net = gas.net_rates_of_progress  # kmol/m³/s

# Species production rates
R = gas.net_production_rates      # kmol/m³/s
R_H2 = R[gas.species_index('H2')]
```

**Reaction information:**
```python
# Number of reactions
n_reactions = gas.n_reactions

# Reaction equations
for i in range(gas.n_reactions):
    eq = gas.reaction_equation(i)
    print(f"Reaction {i}: {eq}")
    print(f"  Forward rate: {w_f[i]:.2e} kmol/m³/s")
    print(f"  Reverse rate: {w_r[i]:.2e} kmol/m³/s")
```

**Equilibrium constants:**
```python
# Equilibrium constants
K = gas.equilibrium_constants

# Third-body efficiencies
eff = gas.third_body_efficiencies
```

### 5. Transport Properties

**Transport model:**
```python
# Create phase with transport
gas = ct.Solution('gri30.yaml', 'gas', transport_model='Mix')
gas.TPX = 300, 101325, 'N2:0.79, O2:0.21'

# Transport properties
D_mix = gas.mix_diff_coeffs      # m²/s (mixture-averaged)
D_bin = gas.binary_diff_coeffs    # m²/s (binary)
lambda_mix = gas.thermal_conductivity  # W/m/K
viscosity = gas.viscosity          # Pa·s
```

**Species transport:**
```python
# Species-specific properties
D_H2 = gas.mix_diff_coeffs[gas.species_index('H2')]
lambda_H2 = gas.thermal_conductivity[gas.species_index('H2')]
```

## Reactor Models

### Ideal Gas Reactors

**Constant pressure reactor:**
```python
# See references/reactors.md for complete details
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 1500, 101325, 'H2:2, O2:1'

r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

# Advance
sim.advance(1e-3)

print(f"Temperature: {gas.T} K")
print(f"Pressure: {gas.P} Pa")
```

**Constant volume reactor:**
```python
r = ct.IdealGasConstVolumeReactor(gas)
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

**Ideal gas reactor (general):**
```python
r = ct.IdealGasReactor(gas)
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

### Flow Reactors

**Plug flow reactor:**
```python
# 1D steady flow
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'H2:2, O2:1'

area = 0.01  # m²
r = ct.FlowReactor(gas, area=area)
sim = ct.ReactorNet([r])

# Advance to steady state
sim.set_initial_time(0.0)
sim.advance_to_steady_state(max_steps=1000)
```

**Reservoir:**
```python
# Infinite reservoir
r1 = ct.Reservoir(gas)
r2 = ct.FlowReactor(gas, area=0.01)

# Connect reactors
sim = ct.ReactorNet([r1, r2])
sim.set_initial_time(0.0)
sim.advance(1e-3)
```

### Wall Reactors

**Adiabatic wall:**
```python
r = ct.Wall(gas, A=0.01)
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

**Heat transfer wall:**
```python
# With heat transfer coefficient
r = ct.Wall(gas, A=0.01, U=100.0)  # W/m²/K
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

**Diabatic wall with mass transfer:**
```python
# Mass transfer to surface phase
surf = ct.Interface('surface.yaml', 'Pt_surf', [gas])
r = ct.Wall(gas, A=0.01, left=surf)
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

### Surface Reactors

**Surface reactor:**
```python
# See references/surface_reactions.md for complete details
gas = ct.Solution('gri30.yaml', 'gas')
surf = ct.Interface('surface.yaml', 'Pt_surf', [gas])

r = ct.IdealGasConstPressureReactor(gas)
rsurf = ct.SurfaceReactor(surf, A=0.01)

sim = ct.ReactorNet([r, rsurf])
sim.advance(1e-3)
```

## 1D Flame Models

### Free Flame

```python
# See references/flames.md for complete implementation
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'H2:2, O2:1'

# Create free flame
f = ct.FreeFlame('gri30.yaml', width=0.02)
f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)

# Solve
f.solve(loglevel=0, refine_grid=True)

# Output
print(f"Flame speed: {f.u[0]:.2f} m/s")
print(f"Max temperature: {f.T.max():.1f} K")
```

### Counterflow Flame

```python
# Stabilized flame
f = ct.CounterflowFlame('gri30.yaml', width=0.02)
f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)

f.solve(loglevel=0, refine_grid=True)
```

### Burner Stabilized Flame

```python
# Burner-stabilized flame
f = ct.BurnerFlame('gri30.yaml', width=0.02)
f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)

f.solve(loglevel=0, refine_grid=True)
```

## Sensitivity Analysis

```python
# Rate-of-production sensitivity
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 1500, 101325, 'H2:2, O2:1'

r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

# Enable sensitivity
sim.algebraic_sensitivity_on()

# Advance
sim.advance(1e-3)

# Get sensitivity coefficients
sens = sim.sensitivity()

# Analyze important reactions
for i in range(gas.n_reactions):
    if abs(sens[i]) > 0.01:
        print(f"Reaction {i}: {gas.reaction_equation(i)}")
        print(f"  Sensitivity: {sens[i]:.4f}")
```

## Path Analysis

**Reaction path analysis:**
```python
# See references/path_analysis.md for complete implementation
# Analyze dominant reaction pathways
```

## Equilibrium Calculations

**Chemical equilibrium:**
```python
# Minimize Gibbs free energy
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 2000, 101325, 'H2:2, O2:1'

# Equilibrate
gas.equilibrate('HP', solver='gibbs')

print(f"Equilibrium composition: {gas.X}")
print(f"Equilibrium temperature: {gas.T} K")
```

**Phase equilibrium:**
```python
# Multiphase equilibrium
gas = ct.Solution('gri30.yaml', 'gas')
liquid = ct.Solution('liquid.yaml', 'liquid')

# Create phase list
phases = [gas, liquid]

# Equilibrate
ct.equilibrate(phases, 'HP', 101325)
```

## Best Practices

### 1. Phase Selection
- Use appropriate phase model (ideal-gas, Redlich-Kwong, etc.)
- Include transport properties for accurate diffusion
- Use surface phases for heterogeneous chemistry

### 2. Reactor Selection
- **Batch reactors**: IdealGasConstPressureReactor, IdealGasConstVolumeReactor
- **Flow reactors**: FlowReactor for steady-state
- **Flame models**: FreeFlame, CounterflowFlame for flames
- **Surface reactors**: SurfaceReactor for catalysis

### 3. Numerical Stability
- Use appropriate time steps
- Enable grid refinement for flames
- Check for stiffness (use CVODE if needed)
- Monitor conservation (mass, energy)

### 4. Performance
- Use compiled mechanisms (CTI) for faster loading
- Enable sensitivity analysis selectively
- Use appropriate solvers for stiff problems
- Consider parallel execution for large problems

### 5. Units
- Cantera uses SI units (K, Pa, J, mol, kg, m, s)
- Be consistent with units throughout
- Convert input/output as needed

## Resources

### Scripts

**`scripts/template_thermo.py`**
Basic thermodynamic calculations template.

**`scripts/template_reactor.py`**
Simple batch reactor simulation template.

**`scripts/template_flame.py`**
1D flame simulation template.

**`scripts/template_equilibrium.py`**
Chemical equilibrium calculation template.

### References

- **`references/thermodynamics.md`** - Thermodynamic properties and phase equilibria
- **`references/kinetics.md`** - Chemical kinetics and reaction mechanisms
- **`references/transport.md`** - Transport properties and diffusion
- **`references/reactors.md`** - Reactor models and configurations
- **`references/flames.md`** - 1D flame models and combustion
- **`references/surface_reactions.md`** - Heterogeneous chemistry and catalysis
- **`references/equilibrium.md`** - Chemical and phase equilibrium
- **`references/sensitivity.md`** - Sensitivity and path analysis
- **`references/input_files.md`** - Creating YAML input files

## Common Issues

**Mechanism not found:**
- Check YAML file path
- Verify phase name in YAML
- Ensure species are defined

**Numerical instability:**
- Reduce time step
- Use stiff solver
- Check for extreme temperatures
- Enable grid refinement

**Slow convergence:**
- Use appropriate solver
- Check mechanism size
- Enable sensitivity analysis
- Consider mechanism reduction

**Conservation errors:**
- Check element conservation
- Verify solver tolerances
- Monitor mass/energy balance
- Check for numerical errors

## Additional Resources

- Official documentation: https://cantera.org/
- Python API: https://cantera.org/stable/python/index.html
- Examples: https://cantera.org/stable/examples/python/index.html
- GitHub repository: https://github.com/Cantera/cantera
- User group: https://groups.google.com/g/cantera-users
