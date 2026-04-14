# Transport Properties in Cantera

Complete guide to transport properties and multicomponent diffusion.

## Transport Models

### Available Models

```python
import cantera as ct

# Mixture-averaged transport
gas = ct.Solution('gri30.yaml', 'gas', transport_model='Mix')

# Unity Lewis numbers
gas = ct.Solution('gri30.yaml', 'gas', transport_model='UnityLewis')

# Multicomponent transport
gas = ct.Solution('gri30.yaml', 'gas', transport_model='Multi')

# Ion transport (for plasmas)
gas = ct.Solution('plasma.yaml', 'gas', transport_model='Ion')
```

### Model Characteristics

**Mix (Mixture-averaged):**
- Uses mixture-averaged properties
- Good for most gas-phase systems
- Reasonable accuracy
- Fast computation

**UnityLewis:**
- Assumes all Lewis numbers = 1
- Approximate but fast
- Good for preliminary calculations

**Multi (Multicomponent):**
- Rigorous multicomponent transport
- Most accurate
- Slower computation
- Required for accurate diffusion

**Ion:**
- For plasma systems
- Includes ion transport
- Charged species transport

## Transport Properties

### Mixture Properties

```python
import cantera as ct

# Create phase with transport
gas = ct.Solution('gri30.yaml', 'gas', transport_model='Multi')
gas.TPX = 300, 101325, 'N2:0.79, O2:0.21'

# Thermal conductivity (W/m/K)
lambda_mix = gas.thermal_conductivity
print(f"Thermal conductivity: {lambda_mix:.6f} W/m/K")

# Viscosity (Pa·s)
viscosity = gas.viscosity
print(f"Viscosity: {viscosity:.6e} Pa·s")

# Density (kg/m³)
density = gas.density
print(f"Density: {density:.6f} kg/m³")

# Mean molecular weight
mw = gas.mean_molecular_weight
printmas(f"Mean MW: {mw:.4f} kg/kmol")
```

### Diffusion Coefficients

**Mixture-averaged diffusion:**
```python
# Mixture-averaged diffusion coefficients (m²/s)
D_mix = gas.mix_diff_coeffs
print(f"Mixture diffusion: {D_mix}")

# Specific species
D_H2 = D_mix[gas.species_index('H2')]
print(f"H2 diffusion: {D_H2:.6f} m²/s")
```

**Binary diffusion coefficients:**
```python
# Binary diffusion coefficients (m²/s)
D_bin = gas.binary_diff_coeffs
print(f"Binary diffusion: {D_bin}")

# H2 in N2
D_H2_N2 = D_bin[gas.species_index('H2'), gas.species_index('N2')]
print(f"H2 in N2: {D_H2_N2:.6f} m²/s")
```

### Thermal Diffusion Ratios

```python
# Thermal diffusion ratios
D_T = gas.thermal_diff_coeffs
print(f"Thermal diffusion ratios: {D_T}")

# Soret effect
D_T_H2 = D_T[gas.species_index('H2')]
print(f"H2 thermal diffusion ratio: {D_T_H2:.6f}")
```

## Species Transport Properties

### Individual Species Properties

```python
# Get species object
species_H2 = gas.species('H2')

# Species properties
print(f"Species: {species_H2.name}")
print(f"Molecular weight: {species_H2.weight} kg/kmol")
print(f"Charge: {species_H2.charge}")
print(f"Size: {species_H2.size} m")
```

### Species Transport Data

```python
# Transport data for species
for i in range(gas.n_species):
    species_name = gas.species_name(i)
    
    # Diffusion coefficient
    D_i = gas.mix_diff_coeffs[i]
    
    # Thermal diffusion ratio
    D_T_i = gas.thermal_diff_coeffs[i]
    
    print(f"{species_name}: D = {D_i:.6e} m²/s, D_T = {D_T_i:.6f}")
```

## Diffusion Velocities

### Multicomponent Diffusion

```python
# Concentration gradients (kmol/m⁴)
grad_X = np.array([0.1, 0.0, 0.0])  # Gradient in x-direction

# Diffusion velocities (m/s)
V_diff = gas.mix_diff_coeffs * grad_X
print(f"Diffusion velocities: {V_diff} m/s")
```

### Thermal Diffusion

```python
# Temperature gradient (K/m)
grad_T = np.array([100.0, 0.0, 0.0])

# Thermal diffusion velocities
V_thermal = gas.thermal_diff_coeffs * grad_T
print(f"Thermal diffusion velocities: {V_thermal} m/s")
```

### Total Diffusion Velocity

```python
# Total diffusion velocity (m/s)
V_total = V_diff + V_thermal
print(f"Total diffusion velocity: {V_total} m/s")
```

## Transport in Reactors

### Transport in Flow Reactors

```python
# See references/reactors.md for complete implementation
# Transport automatically included in flow reactors

gas = ct.Solution('gri30.yaml', 'gas', transport_model='Multi')
gas.TPX = 300, 101325, 'H2:2, O2:1'

# Flow reactor with transport
area = 0.01  # m²
r = ct.FlowReactor(gas, area=area)
sim = ct.ReactorNet([r])

sim.advance_to_steady_state()
```

### Transport in 1D Flames

```python
# See references/flames.md for complete implementation
# Transport required for accurate flame structure

f = ct.FreeFlame('gri30.yaml', width=0.02, transport_model='Multi')
f.solve(loglevel=0, refine_grid=True)
```

## Transport Coefficient Calculation

### Chapman-Enskog Theory

```python
# Cantera uses Chapman-Enskog theory for gas-phase transport
# Automatically calculated from species properties

# Check if transport is available
if gas.transport_model:
    print("Transport model: " + gas.transport_model)
    print("Thermal conductivity: " + str(gas.thermal_conductivity))
    print("Viscosity: " + str(gas.viscosity))
else:
    print("No transport model available")
```

### Wilke's Formula

```python
# Used for binary diffusion coefficients
# See Cantera documentation for details
```

## Common Applications

### Flame Speed Calculation

```python
# Transport required for accurate flame speed
f = ct.FreeFlame('gri30.yaml', width=0.02, transport_model='Multi')
f.solve(loglevel=0, refine_grid=True)

print(f"Laminar flame speed: {f.u[0]:.2f} m/s")
```

### Ignition Delay

```python
# Transport affects ignition delay
# Important for autoignition modeling
```

### Diffusion Flames

```python
# Counterflow diffusion flame
f = ct.CounterflowFlame('gri30.yaml', width=0.02, transport_model='Multi')
f.solve(loglevel=0, refine_grid=True)
```

### Species Separation

```python
# Isotope separation
# Requires accurate multicomponent transport
```

## Numerical Considerations

### Transport Model Selection

**Guidelines:**
- Use 'Mix' for most applications
- Use 'Multi' for high accuracy
- Use 'UnityLewis' for fast preliminary calculations
- Use 'Ion' for plasma systems

### Accuracy vs. Speed

**Trade-offs:**
- 'Mix': Good balance
- 'Multi': Most accurate, slower
- 'UnityLewis': Fast, less accurate

### Temperature Dependence

```python
# Transport properties are temperature-dependent
temperatures = np.linspace(300, 2000, 100)

for T in temperatures:
    gas.T = T
    lambda_T = gas.thermal_conductivity
    visc_T = gas.viscosity
    # Record properties
```

### Pressure Dependence

```python
# Transport properties are pressure-dependent
pressures = np.logspace(1e3, 1e6, 100)

for P in pressures:
    gas.P = P
    lambda_P = gas.thermal_conductivity
    visc_P = gas.viscosity
    # Record properties
```

## Troubleshooting

### Transport Not Available

**Issue:** Transport properties not available

**Solutions:**
- Check input file includes transport data
- Verify transport model is specified
- Ensure species have transport data

### Negative Diffusion Coefficients

**Issue:** Unphysical negative diffusion

**Solutions:**
- Check temperature range
- Verify transport data
- Use appropriate transport model
- Check for numerical errors

### Poor Convergence

**Issue:** Slow convergence with transport

**Solutions:**
- Use simpler transport model
- Check transport data accuracy
- Reduce grid resolution
- Improve initial guess

### Inconsistent Results

**Issue:** Results differ between models

**Solutions:**
- Verify transport data consistency
- Check temperature/pressure ranges
- Use appropriate model for application
- Compare with experimental data

## Advanced Topics

### Custom Transport Models

```python
# Define custom transport model
# See Cantera documentation for details
```

### High-Pressure Transport

```python
# High-pressure corrections
# Important for combustion at high pressure
```

### Plasma Transport

```python
# Ion and electron transport
# Charged species interactions
```

### Surface Transport

```python
# Surface diffusion
# See references/surface_reactions.md
```

## Resources

- Cantera transport reference: https://cantera.org/stable/reference/transport/index.html
- Transport models: https://cantera.org/stable/python/html/cantera.html#cantera.Solution
- Transport examples: https://cantera.org/stable/examples/python/transport/
