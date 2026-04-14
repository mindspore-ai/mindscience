# Chemical and Phase Equilibrium

Complete guide to chemical and phase equilibrium calculations.

## Chemical Equilibrium

### Single-Phase Equilibrium

```python
import cantera as ct

# Create gas phase
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 2000, 101325, 'H2:2, O2:1'

# Equilibrate at constant T, P
gas.equilibrate('HP', solver='gibbs')

print(f"Equilibrium composition: {gas.X}")
print(f"Equilibrium temperature: {gas.T} K")
```

### Adiabatic Equilibrium

```python
# Constant enthalpy and pressure
gas.equilibrate('HP', solver='gibbs')
```

### Isochoric Equilibrium

```python
# Constant internal energy and density
gas.equilibrate('UV', solver='gibbs')
```

### Isothermal Equilibrium

```python
# Constant temperature and pressure
gas.equilibrate('TP', solver='gibbs')
```

### Isobaric Equilibrium

```python
# Constant pressure and enthalpy
gas.equilibrate('HP', solver='gibbs')
```

## Equilibrium Solvers

### Available Solvers

```python
# Available solvers
solvers = ['gibbs', 'vcs', 'newton']

# Equilibrate with specific solver
gas.equilibrate('HP', solver='newton', max_steps=100, tol=1e-10)
```

### Solver Options

```python
# Equilibrate with options
gas.equilibrate(
    'HP',
    solver='gibbs',
    max_steps=1000,     # Maximum iterations
    tol=1e-10,           # Tolerance
    max_step_size=0.1,    # Maximum step size
    log_level=0          # Logging level
)
```

### Convergence Issues

```python
# Try different solvers if convergence fails
for solver in ['gibbs', 'vcs', 'newton']:
    try:
        gas.equilibrate('HP', solver=solver)
        print(f"Converged with {solver}")
        break
    except:
        print(f"Failed with {solver}")
```

## Multi-Phase Equilibrium

### Vapor-Liquid Equilibrium

```python
# Create gas and liquid phases
gas = ct.Solution('gas.yaml', 'gas')
liquid = ct.Solution('liquid.yaml', 'liquid')

# Set initial conditions
gas.TPX = 300, 101325, 'H2O:0.5'
liquid.TPX = 300, 101325, 'H2O:0.5'

# Create phase list
phases = [gas, liquid]

# Equilibrate
ct.equilibrate(phases, 'HP', 101325)

print(f"Gas phase: {gas.X}")
print(f"Liquid phase: {liquid.X}")
```

### Condensed Phase Equilibrium

```python
# Solid-gas equilibrium
solid = ct.Solution('solid.yaml', 'solid')
gas = ct.Solution('gas.yaml', 'gas')

phases = [solid, gas]
ct.equilibrate(phases, 'HP', 101325)
```

### Multi-Component Equilibrium

```python
# Multiple phases with multiple components
gas = ct.Solution('gas.yaml', 'gas')
liquid1 = ct.Solution('liquid1.yaml', 'liquid')
liquid2 = ct.Solution('liquid2.yaml', 'liquid')

phases = [gas, liquid1, liquid2]
ct.equilibrate(phases, 'HP', 101325)
```

## Equilibrium Constants

### Calculating K

```python
# Calculate equilibrium constants
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 1500, 101325, 'H2:0.5, O2:0.5, H2O:0.0'

# Equilibrate
gas.equilibrate('HP', solver='gibbs')

# Get equilibrium constants
K = gas.equilibrium_constants

for i in range(gas.n_reactions):
    if gas.reaction(i).reversible:
        print(f"Reaction {i}: K = {K[i]:.6e}")
```

### Temperature Dependence

```python
# Calculate K at different temperatures
temperatures = np.linspace(300, 2000, 100)

for T in temperatures:
    gas.T = T
    gas.equilibrate('HP', solver='gibbs')
    K = gas.equilibrium_constants
    # Record K values
```

### van't Hoff Equation

```python
# Calculate enthalpy of reaction
gas1 = ct.Solution('gri30.yaml', 'gas')
gas1.TPX = 300, 101325, 'H2:2, O2:1'

h_reactants = gas1.enthalpy_mole

# Equilibrate
gas2 = ct.Solution('gri30.yaml', 'gas')
gas2.TPX = 300, 101325, 'H2:2, O2:1'
gas2.equilibrate('HP', solver='gibbs')

h_products = gas2.enthalpy_mole
delta_H = h_products - h_reactants

print(f"Heat of reaction: {delta_H} J/mol")
```

## Phase Diagrams

### T-X Diagram

```python
# Temperature-composition diagram
temperatures = np.linspace(300, 1000, 100)
X_H2O = np.zeros(len(temperatures))

for i, T in enumerate(temperatures):
    gas.T = T
    liquid.T = T
    
    # Equilibrate
    ct.equilibrate([gas, liquid], 'HP', 101325)
    
    # Record composition
    X_H2O[i] = liquid.X[liquid.species_index('H2O')]

# Plot T-X diagram
import matplotlib.pyplot as plt
plt.plot(X_H2O, temperatures)
plt.xlabel('H2O mole fraction')
plt.ylabel('Temperature (K)')
plt.title('T-X Diagram')
plt.show()
```

### P-T Diagram

```python
# Pressure-temperature diagram
temperatures = np.linspace(300, 1000, 100)
pressures = np.zeros(len(temperatures))

for i, T in enumerate(temperatures):
    gas.T = T
    liquid.T = T
    
    # Equilibrate
    ct.equilibrate([gas, liquid], 'HP', 101325)
    
    # Record pressure
    pressures[i] = gas.P

# Plot P-T diagram
plt.plot(temperatures, pressures)
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (Pa)')
plt.title('P-T Diagram')
plt.show()
```

## Common Applications

### Combustion Equilibrium

```python
# Adiabatic flame temperature
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'CH4:1, O2:2'

# Equilibrate
gas.equilibrate('HP', solver='gibbs')

T_ad = gas.T
print(f"Adiabatic flame temperature: {T_ad} K")
```

### Equilibrium Composition

```python
# Find equilibrium composition
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 1500, 101325, 'H2:0.5, O2:0.5, H2O:0.0'

gas.equilibrate('HP', solver='gibbs')

print(f"Products: {gas.X}")
```

### Phase Stability

```python
# Determine stable phase
gas = ct.Solution('gas.yaml', 'gas')
liquid = ct.Solution('liquid.yaml', 'liquid')

# Compare Gibbs energies
g_gas = gas.gibbs_mole
g_liquid = liquid.gibbs_mole

if g_gas < g_liquid:
    print("Gas phase is stable")
else:
    print("Liquid phase is stable")
```

## Numerical Considerations

### Convergence

**Issue:** Equilibrium solver fails

**Solutions:**
- Try different solver
- Improve initial guess
- Reduce max_step_size
- Check thermodynamic data

### Accuracy

**Improve accuracy:**
- Use high-precision thermodynamic data
- Check thermodynamic consistency
- Verify phase model
- Use appropriate solver

### Performance

**Speed up calculations:**
- Use compiled mechanisms (CTI)
- Cache equilibrium calculations
- Use appropriate solver
- Consider mechanism reduction

## Troubleshooting

### Invalid State

**Issue:** Cannot equilibrate

**Solutions:**
- Check phase model
- Verify composition
- Check temperature/pressure ranges
- Ensure species exist

### Solver Failure

**Issue:** Solver fails to converge

**Solutions:**
- Try different solver
- Improve initial guess
- Reduce max_step_size
- Check thermodynamic data

### Phase Issues

**Issue:** Phase not stable

**Solutions:**
- Check Gibbs energies
- Verify phase model
- Check thermodynamic data
- Consult mechanism documentation

## Advanced Topics

### Custom Equilibrium

```python
# Define custom equilibrium problem
# See Cantera documentation for details
```

### Kinetic vs Equilibrium

```python
# Compare kinetic and equilibrium results
# Important for validating mechanisms
```

### Phase Stability Analysis

```python
# Analyze phase stability
# Determine phase boundaries
```

## Resources

- Cantera equilibrium reference: https://cantera.org/stable/reference/thermo/equilibrium.html
- Equilibrium examples: https://cantera.org/stable/examples/python/thermo/
- Phase diagrams: https://cantera.org/stable/examples/python/thermo/phaseDiagram.py
