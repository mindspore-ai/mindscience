# Sensitivity and Path Analysis

Complete guide to sensitivity analysis and reaction pathway analysis.

## Sensitivity Analysis

### Rate-of-Production Sensitivity

```python
import cantera as ct

# Create gas phase
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 1500, 101325, 'H2:2, O2:1'

# Create reactor
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

# Enable sensitivity
sim.algebraic_sensitivity_on()

# Advance
sim.advance(1e-3)

# Get sensitivity coefficients
sens = sim.sensitivity()

# Analyze
for i in range(gas.n_species):
    for j in range(gas.n_reactions):
        if abs(sens[i, j]) > 0.01:
            print(f"Species {gas.species_name(i)} sensitive to reaction {j}")
            print(f"  Reaction: {gas.reaction_equation(j)}")
            print(f"  Sensitivity: {sens[i, j]:.4f}")
```

### Species Sensitivity

```python
# Get sensitivity for specific species
sens_H2 = sens[gas.species_index('H2'), :]

# Find most sensitive reactions
for j in range(gas.n_reactions):
    if abs(sens_H2[j]) > 0.1:
        print(f"H2 sensitive to reaction {j}: {sens_H2[j]:.4f}")
```

### Reaction Sensitivity

```python
# Get sensitivity for specific reaction
sens_rxn = sens[:, reaction_index]

# Find most sensitive species
for i in range(gas.n_species):
    if abs(sens_rxn[i]) > 0.1:
        print(f"Reaction {reaction_index} affects {gas.species_name(i)}")
        print(f"  Sensitivity: {sens_rxn[i]:.4f}")
```

## Reaction Pathway Analysis

### Dominant Reactions

```python
# Identify reactions with highest rates
gas.TPX = 1500, 101325, 'H2:2, O2:1'

# Reaction rates
w_f = gas.forward_rates_of_progress
w_r = gas.reverse_rates_of_progress
w_net = w_f - w_r

# Find dominant reactions
max_rate = max(abs(w_net))

for j in range(gas.n_reactions):
    if abs(w_net[j]) > 0.1 * max_rate:
        print(f"Dominant reaction {j}: {gas.reaction_equation(j)}")
        print(f"  Net rate: {w_net[j]:.6e} kmol/m³/s")
```

### Reaction Flux Analysis

```python
# See Cantera path analysis examples
# Analyze reaction pathways and fluxes
```

### Rate-of-Production Analysis (ROPA)

```python
# ROPA: rate-of-production analysis
# Analyze which reactions contribute most to species production

R = gas.net_production_rates

for i in range(gas.n_species):
    species_name = gas.species_name(i)
    production_rate = R[i]
    
    if abs(production_rate) > 1e-10:
        print(f"{species_name}: {production_rate:.6e} kmol/m³/s")
        
        # Find contributing reactions
        for j in range(gas.n_reactions):
            nu = gas.reactant_stoich_coeffs[:, j]
            nu_p = gas.product_stoich_coeffs[:, j]
            
            contribution = (nu_p[i] - nu[i]) * w_net[j]
            
            if abs(contribution) > 0.1 * abs(production_rate):
                print(f"  Reaction {j}: {contribution:.6e} kmol/m³/s")
```

## Eigenvalue Analysis

### Jacobian Eigenvalues

```python
# Compute eigenvalues of Jacobian
# See Cantera eigenvalue examples
# Identify stiff modes
```

### Time Scale Analysis

```python
# Analyze time scales of reactions
# Identify fast and slow reactions
```

## Mechanism Reduction

### Eliminate Unimportant Species

```python
# Identify species with low sensitivity
unimportant_species = []

for i in range(gas.n_species):
    species_sens = np.max(np.abs(sens[i, :]))
    
    if species_sens < 0.01:
        unimportant_species.append(gas.species_name(i))

print(f"Unimportant species: {unimportant_species}")
```

### Eliminate Unimportant Reactions

```python
# Identify reactions with low sensitivity
unimportant_reactions = []

for j in range(gas.n_reactions):
    reaction_sens = np.max(np.abs(sens[:, j]))
    
    if reaction_sens < 0.01:
        unimportant_reactions.append(j)

print(f"Unimportant reactions: {unimportant_reactions}")
```

### QSSA (Quasi-Steady-State Approximation)

```python
# Identify QSS species
# Species with fast consumption/production

# See Cantera QSSA examples
```

## Common Applications

### Combustion Sensitivity

```python
# High-temperature combustion
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 2000, 101325, 'CH4:1, O2:2'

r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

sim.algebraic_sensitivity_on()
sim.advance(1e-3)

sens = sim.sensitivity()

# Analyze key species
key_species = ['H2', 'O2', 'H2O', 'OH', 'O', 'H']

for species in key_species:
    if species in gas.species_names:
        sens_species = sens[gas.species_index(species), :]
        max_sens = np.max(np.abs(sens_species))
        print(f"{species}: max sensitivity = {max_sens:.4f}")
```

### Ignition Sensitivity

```python
# Analyze sensitivity during ignition
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 1000, 101325, 'H2:2, O2:1'

r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

sim.algebraic_sensitivity_on()

for step in range(100):
    sim.advance(1e-4)
    
    if step % 10 == 0:
        sens = sim.sensitivity()
        print(f"Time: {sim.time:.6f} s, T: {gas.T:.1f} K")
```

### Catalysis Sensitivity

```python
# Surface reaction sensitivity
# See references/surface_reactions.md
```

## Numerical Considerations

### Sensitivity Calculation

**Performance:**
- Enable selectively (not for all calculations)
- Use for mechanism reduction
- Analyze at relevant conditions

**Accuracy:**
- Check convergence
- Verify sensitivity magnitudes
- Compare with analytical results

### Pathway Analysis

**Guidelines:**
- Use appropriate time scales
- Check for steady-state
- Verify mass balance
- Analyze multiple conditions

## Troubleshooting

### Sensitivity Not Available

**Issue:** Sensitivity coefficients not computed

**Solutions:**
- Enable algebraic sensitivity
- Check reactor type
- Verify solver
- Consult documentation

### Zero Sensitivity

**Issue:** All sensitivities are zero

**Solutions:**
- Check time step
- Verify reaction rates
- Check state variables
- Ensure sensitivity is enabled

### Unphysical Sensitivities

**Issue:** Sensitivity values are unphysical

**Solutions:**
- Check mechanism consistency
- Verify thermodynamic data
- Check numerical stability
- Reduce time step

## Advanced Topics

### Local Sensitivity

```python
# Spatial sensitivity in flames
# See Cantera flame sensitivity examples
```

### Global Sensitivity

```python
# Global sensitivity analysis
# Analyze overall system behavior
```

### Time-Dependent Sensitivity

```python
# Track sensitivity over time
# Identify changing dominant pathways
```

## Resources

- Cantera sensitivity reference: https://cantera.org/stable/reference/kinetics/sensitivity.html
- Sensitivity examples: https://cantera.org/stable/examples/python/kinetics/
- Path analysis examples: https://cantera.org/stable/examples/python/kinetics/
