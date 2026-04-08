# 1D Flame Models in Cantera

Complete guide to 1D flame models and combustion.

## Free Flames

### Basic Free Flame

```python
import cantera as ct

# Create gas phase
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'H2:2, O2:1'

# Create free flame
f = ct.FreeFlame('gri30.yaml', width=0.02)

# Set refine criteria
f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)

# Solve
f.solve(loglevel=0, refine_grid=True)

# Output
print(f"Flame speed: {f.u[0]:.2f} m/s")
print(f"Max temperature: {f.T.max():.1f} K")
print(f"Grid points: {len(f.T)}")
```

### Free Flame with Transport

```python
# Create flame with transport model
f = ct.FreeFlame('gri30.yaml', width=0.02, transport_model='Mix')

f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)
f.solve(loglevel=0, refine_grid=True)
```

### Free Flame with Specific Grid

```python
# Create flame with specified grid
f = ct.FreeFlame('gri30.yaml', width=0.02)

# Set initial grid
f.set_initial_grid(dom=ct.Domain1D(flame.grid))

# Solve
f.solve(loglevel=0, refine_grid=False)
```

## Counterflow Flames

### Basic Counterflow Flame

```python
import cantera as ct

# Create gas phase
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'H2:2, O2:1'

# Create counterflow flame
f = ct.CounterflowFlame('gri30.yaml', width=0.02)

# Set refine criteria
f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)

# Solve
f.solve(loglevel=0, refine_grid=True)

print(f"Flame speed: {f.u[0]:.2f} m/s")
print(f"Max temperature: {f.T.max():.1f} K")
```

### Counterflow Flame with Strain Rate

```python
# Set strain rate
f = ct.CounterflowFlame('gri30.yaml', width=0.02)
f.strain_rate = 100.0  # 1/s

f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)
f.solve(loglevel=0, refine_grid=True)
```

## Burner-Stabilized Flames

### Basic Burner Flame

```python
import cantera as ct

# Create gas phase
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'H2:2, O2:1'

# Create burner-stabilized flame
f = ct.BurnerFlame('gri30.yaml', width=0.02)

# Set refine criteria
f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)

# Solve
f.solve(loglevel=0, refine_grid=True)

print(f"Flame speed: {f.u[0]:.2f} m/s")
print(f"Max temperature: {f.T.max():.1f} K")
```

### Burner Flame with Mass Flow

```python
# Set mass flow rate
f = ct.BurnerFlame('gri30.yaml', width=0.02)
f.mdot = 0.01  # kg/m²/s

f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)
f.solve(loglevel=0, refine_grid=True)
```

## Flame Properties

### Flame Speed

```python
# Laminar flame speed
S_L = f.u[0]
print(f"Laminar flame speed: {S_L:.2f} m/s")

# Flame speed at different positions
S = f.u
print(f"Flame speed profile: {S}")
```

### Temperature Profile

```python
# Temperature profile
T = f.T
print(f"Temperature: {T}")

# Maximum temperature
T_max = f.T.max()
print(f"Max temperature: {T_max:.1f} K")

# Flame thickness
delta_T = T_max - T.min()
print(f"Temperature rise: {delta_T:.1f} K")
```

### Species Profiles

```python
# Species profiles
X = f.X  # Mole fractions
Y = f.Y  # Mass fractions

# Specific species
X_H2 = X[gas.species_index('H2')]
Y_H2 = Y[gas.species_index('H2')]

print(f"H2 mole fraction: {X_H2}")
print(f"H2 mass fraction: {Y_H2}")
```

### Heat Release Rate

```python
# Heat release rate (W/m³)
q = f.heat_release_rate

print(f"Heat release rate: {q}")
```

## Grid Refinement

### Refine Criteria

```python
# Set refine criteria
f.set_refine_criteria(
    ratio=3.0,      # Max cell size ratio
    slope=0.1,      # Max slope
    curve=0.1,      # Max curvature
    prune=0.0        # Prune threshold
)
```

### Adaptive Grid

```python
# Enable adaptive refinement
f.solve(loglevel=0, refine_grid=True)

# Check grid quality
print(f"Grid points: {len(f.T)}")
print(f"Min grid spacing: {min(f.grid):.6f} m")
print(f"Max grid spacing: {max(f.grid):.6f} m")
```

### Manual Grid Control

```python
# Disable automatic refinement
f.solve(loglevel=0, refine_grid=False)

# Manually refine
f.refine(ratio=3.0, slope=0.1, curve=0.1)
```

## Boundary Conditions

### Inlet Conditions

```python
# Set inlet temperature
f.inlet.T = 300  # K

# Set inlet composition
f.inlet.X = {'H2': 0.5, 'O2': 0.5}

# Set inlet pressure
f.inlet.P = 101325  # Pa
```

### Outlet Conditions

```python
# Outlet is typically zero-gradient
# No explicit setting needed
```

## Flame Analysis

### Flame Thickness

```python
# Thermal thickness
delta_T = f.T.max() - f.T.min()
delta_x = f.grid[-1] - f.grid[0]

# Characteristic flame thickness
delta_flame = delta_x / delta_T
print(f"Flame thickness: {delta_flame:.6f} m")
```

### Flame Position

```python
# Find position of max temperature
T_max = f.T.max()
pos_max = f.grid[f.T.argmax()]

print(f"Flame position: {pos_max:.6f} m")
```

### Species Consumption

```python
# Calculate species consumption
X_inlet = f.inlet.X
X_outlet = f.outlet.X

for species in gas.species_names:
    delta_X = X_inlet[gas.species_index(species)] - X_outlet[gas.species_index(species)]
    print(f"{species}: {delta_X:.6f}")
```

## Common Applications

### Laminar Premixed Flames

```python
# Standard premixed flame
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'CH4:1, O2:2, N2:7.52')

f = ct.FreeFlame('gri30.yaml', width=0.02)
f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)
f.solve(loglevel=0, refine_grid=True)

print(f"Flame speed: {f.u[0]:.2f} m/s")
```

### Strained Flames

```python
# Counterflow flame with strain
f = ct.CounterflowFlame('gri30.yaml', width=0.02)
f.strain_rate = 100.0  # 1/s

f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)
f.solve(loglevel=0, refine_grid=True)
```

### Burner-Stabilized Flames

```python
# Burner-stabilized flame
f = ct.BurnerFlame('gri30.yaml', width=0.02)
f.mdot = 0.01  # kg/m²/s

f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.1)
f.solve(loglevel=0, refine_grid=True)
```

## Numerical Considerations

### Convergence

**Issue:** Flame solver fails to converge

**Solutions:**
- Improve initial guess
- Adjust refine criteria
- Check transport model
- Verify mechanism

### Grid Quality

**Issue:** Poor grid resolution

**Solutions:**
- Use tighter refine criteria
- Increase max grid points
- Check flame thickness
- Use appropriate transport model

### Transport Model

**Guidelines:**
- Use 'Mix' for most applications
- Use 'Multi' for high accuracy
- Use 'UnityLewis' for fast calculations

### Stiffness

**Issue:** Stiff chemistry

**Solutions:**
- Use appropriate time steps
- Check mechanism size
- Consider mechanism reduction
- Use appropriate solver

## Troubleshooting

### Negative Speed

**Issue:** Negative flame speed

**Solutions:**
- Check boundary conditions
- Verify mechanism
- Improve initial guess
- Check transport model

### Grid Blow-up

**Issue:** Grid becomes too dense

**Solutions:**
- Adjust refine criteria
- Set max grid points
- Use coarser initial grid
- Check flame behavior

### Temperature Errors

**Issue:** Unphysical temperatures

**Solutions:**
- Check energy conservation
- Verify mechanism
- Check boundary conditions
- Use appropriate transport model

## Advanced Topics

### Multi-Component Flames

```python
# Flames with detailed transport
# Use 'Multi' transport model
```

### Extinction Analysis

```python
# Find extinction limit
# Vary strain rate or flow rate
```

### Flame Instability

```python
# Analyze flame stability
# Study critical conditions
```

## Resources

- Cantera 1D flame reference: https://cantera.org/stable/reference/onedim/index.html
- Flame examples: https://cantera.org/stable/examples/python/onedim/
- Flame tutorial: https://cantera.org/stable/userguide/onedim.html
