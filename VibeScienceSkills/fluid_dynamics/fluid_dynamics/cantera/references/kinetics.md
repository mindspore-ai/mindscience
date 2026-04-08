# Chemical Kinetics in Cantera

Complete guide to chemical kinetics and reaction mechanisms.

## Reaction Mechanisms

### Loading Mechanisms

```python
import cantera as ct

# From YAML file
gas = ct.Solution('gri30.yaml', 'gas')

# From CTI file (legacy)
gas = ct.Solution('gri30.cti', 'gas')

# With transport properties
gas = ct.Solution('gri30.yaml', 'gas', transport_model='Mix')
```

### Reaction Information

```python
# Number of reactions
n_reactions = gas.n_reactions
print(f"Number of reactions: {n_reactions}")

# Reaction equations
for i in range(gas.n_reactions):
    eq = gas.reaction_equation(i)
    print(f"Reaction {i}: {eq}")
```

### Reaction Types

**Elementary reactions:**
```python
# Check if reaction is elementary
for i in range(gas.n_reactions):
    reversible = gas.reaction(i).reversible
    print(f"Reaction {i}: {'reversible' if reversible else 'irreversible'}")
```

**Three-body reactions:**
```python
# Check for third-body efficiency
for i in range(gas.n_reactions):
    eff = gas.third_body_efficiencies[i]
    if eff != 1.0:
        print(f"Reaction {i}: third-body efficiency = {eff}")
```

**Fall-off reactions:**
```python
# Check for fall-off reactions
for i in range(gas.n_reactions):
    if gas.reaction(i).type == 'falloff-reaction':
        print(f"Reaction {i}: fall-off reaction")
```

## Reaction Rates

### Forward and Reverse Rates

```python
import cantera as ct

gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 1500, 101325, 'H2:2, O2:1'

# Forward rates (kmol/m³/s)
w_f = gas.forward_rates_of_progress

# Reverse rates (kmol/m³/s)
w_r = gas.reverse_rates_of_progress

# Net rates (kmol/m³/s)
w_net = gas.net_rates_of_progress

print(f"Forward rates: {w_f}")
print(f"Reverse rates: {w_r}")
print(f"Net rates: {w_net}")
```

### Species Production Rates

```python
# Species production rates (kmol/m³/s)
R = gas.net_production_rates

# Production rate for specific species
R_H2 = R[gas.species_index('H2')]
print(f"H2 production rate: {R_H2:.6e} kmol/m³/s")
```

### Reaction Progress Rates

```python
# Progress rate for each reaction
q = gas.net_rates_of_progress

for i in range(gas.n_reactions):
    print(f"Reaction {i}: {q[i]:.6e} kmol/m³/s")
```

## Rate Constants

### Arrhenius Parameters

```python
# Get Arrhenius parameters for each reaction
for i in range(gas.n_reactions):
    rxn = gas.reaction(i)
    
    # Forward rate parameters
    A_f = rxn.rate.pre_exponential_factor
    b_f = rxn.rate.temperature_exponent
    E_f = rxn.rate.activation_energy_R
    
    print(f"Reaction {i}:")
    print(f"  A = {A_f:.6e}")
    print(f"  b = {b_f:.6f}")
    print(f"  E/R = {E_f:.6f} K")
```

### Rate Constants at Temperature

```python
# Calculate rate constants at specific temperature
gas.T = 1500  # K

for i in range(gas.n_reactions):
    rxn = gas.reaction(i)
    
    # Forward rate constant
    k_f = rxn.rate(gas.T)
    
    # Reverse rate constant (if reversible)
    if rxn.reversible:
        k_r = rxn.rate(gas.T, direction=-1)
        print(f"Reaction {i}: k_f = {k_f:.6e}, k_r = {k_r:.6e}")
    else:
        print(f"Reaction {i}: k_f = {k_f:.6e}")
```

### Equilibrium Constants

```python
# Equilibrium constants
K = gas.equilibrium_constants

for i in range(gas.n_reactions):
    if gas.reaction(i).reversible:
        print(f"Reaction {i}: K = {K[i]:.6e}")
```

## Reaction Pathways

### Dominant Reactions

```python
# Identify reactions with highest rates
gas.TPX = 1500, 101325, 'H2:2, O2:1'

q = gas.net_rates_of_progress
max_rate = max(abs(q))

for i in range(gas.n_reactions):
    if abs(q[i]) > 0.1 * max_rate:
        print(f"Dominant reaction {i}: {gas.reaction_equation(i)}")
        print(f"  Rate: {q[i]:.6e} kmol/m³/s")
```

### Reaction Flux Analysis

```python
# See references/path_analysis.md for complete implementation
# Analyze reaction pathways and fluxes
```

### Rate of Production Analysis

```python
# ROPA: rate of production analysis
# Analyze which reactions contribute most to species production

for i in range(gas.n_species):
    species_name = gas.species_name(i)
    production_rate = R[i]
    
    if abs(production_rate) > 1e-10:
        print(f"{species_name}: {production_rate:.6e} kmol/m³/s")
```

## Mechanism Reduction

### Elimination of Unimportant Species

```python
# See references/mechanism_reduction.md for complete implementation
# Remove species that don't significantly affect the system
```

### Reaction Lumpng

```python
# Combine similar reactions
# Improve computational efficiency
```

### Time-Scale Decomposition

```python
# Separate fast and slow reactions
# Use QSSA or similar methods
```

## Sensitivity Analysis

### Rate-of-Production Sensitivity

```python
# See references/sensitivity.md for complete implementation
# Compute sensitivity of species to reaction rates
```

### Eigenvalue Analysis

```python
# Compute eigenvalues of Jacobian
# Identify stiff modes
```

## Common Applications

### Combustion Kinetics

```python
# High-temperature combustion
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 2000, 101325, 'CH4:1, O2:2'

# Reaction rates
w_f = gas.forward_rates_of_progress
print(f"Combustion rates: {w_f}")
```

### Low-Temperature Oxidation

```python
# Low-temperature chemistry
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 600, 101325, 'H2:2, O2:1'

# Important for autoignition
w_f = gas.forward_rates_of_progress
```

### Pyrolysis

```python
# Fuel decomposition
gas = ct.Solution('mech.yaml', 'gas')
gas.TPX = 1000, 101325, 'fuel:1, N2:4')

# Decomposition rates
R = gas.net_production_rates
```

### Catalysis

```python
# Surface reactions
# See references/surface_reactions.md
```

## Numerical Considerations

### Stiffness

**Issue:** Wide range of time scales

**Solutions:**
- Use stiff solvers (CVODE)
- Enable analytical Jacobian
- Consider QSSA for fast species
- Use appropriate time steps

### Accuracy

**Improve accuracy:**
- Use high-precision mechanisms
- Check reaction consistency
- Verify rate constants
- Use appropriate solver tolerances

### Performance

**Speed up calculations:**
- Use compiled mechanisms (CTI)
- Enable sensitivity selectively
- Consider mechanism reduction
- Use appropriate solvers

## Troubleshooting

### Negative Concentrations

**Issue:** Species concentrations become negative

**Solutions:**
- Reduce time step
- Use more stable solver
- Check reaction rates
- Verify mechanism consistency

### Non-Convergence

**Issue:** Solver fails to converge

**Solutions:**
- Improve initial guess
- Reduce time step
- Use different solver
- Check for stiffness

### Slow Reactions

**Issue:** Reactions too slow

**Solutions:**
- Check rate constants
- Verify temperature
- Check pressure effects
- Review mechanism

## Advanced Topics

### Custom Reactions

```python
# Define custom reaction rates
# Modify reaction objects
```

### Pressure Dependence

```python
# Pressure-dependent rate constants
# Fall-off reactions
# Unimolecular fall-off
```

### Third-Body Efficiencies

```python
# Pressure-dependent third-body efficiencies
# High-pressure limit
# Low-pressure limit
```

## Resources

- Cantera kinetics reference: https://cantera.org/stable/reference/kinetics/index.html
- Reaction mechanisms: https://cantera.org/stable/examples/python/onedim/
- Sensitivity analysis: https://cantera.org/stable/examples/python/kinetics/
