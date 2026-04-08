# Mixtures

CoolProp provides high-accuracy mixture property calculations using Helmholtz energy formulations.

## Predefined Mixtures

CoolProp includes several predefined mixtures:

### Air Mixture

**Characteristics**:
- Standard atmospheric composition
- 21% O2, 78% N2, 1% Ar, trace gases
- Valid from 59.15 K to 2000 K

**Usage**:
```python
from CoolProp.CoolProp import PropsSI

# Density at 1 atm, 300 K
rho = PropsSI('D', 'P', 101325.0, 'T', 300.0, 'Air.mix')
```

### Ammonia Mixture

**Characteristics**:
- Common refrigerant mixture
- 20% NH3, 80% N2
- Valid from 143.15 K to 473.15 K

**Usage**:
```python
# Density at 1 atm, 300 K
rho = PropsSI('D', 'P', 101325.0, 'T', 300.0, 'AMARILLO.mix')
```

### Other Predefined Mixtures

**Available mixtures**:
- `EKOFISK.MIX`: EKOFISK mixture
- `Air.mix`: Air mixture
- `Amarillo.mix`: Ammonia mixture
- `Propane-Ethane.mix`: Propane/Ethane mixture

**List all predefined mixtures**:
```python
from CoolProp.CoolProp import CoolProp

CP = CoolProp.CoolProp()
mixtures = CP.get_global_param_string('predefined_mixtures').split(',')
print(mixtures)
```

## User-Defined Mixtures

### Binary Mixtures

**Characteristics**:
- Two-component mixtures
- Binary interaction parameters
- Excess Helmholtz energy terms

**Usage**:
```python
# Propane/Ethane mixture
rho = PropsSI('D', 'P', 101325.0, 'T', 300.0, 
              'HEOS::R32[0.697615]&Ethane[0.302385]')
```

### Ternary Mixtures

**Characteristics**:
- Three-component mixtures
- More complex interaction parameters
- Higher accuracy formulations

**Usage**:
```python
# R32/Ethane/Propane mixture
rho = PropsSI('D', 'P', 101325.0, 'T', 300.0, 
              'HEOS::R32[0.697615]&Ethane[0.302385]&Propane[0.098623]')
```

### Multi-Component Mixtures

**Characteristics**:
- Four or more components
- Complex interaction terms
- Requires good initial guesses

**Usage**:
```python
# Four-component mixture
rho = PropsSI('D', 'P', 101325.0, 'T', 300.0, 
              'COMP1[0.5]&COMP2[0.3]&COMP3[0.15]&COMP4[0.05]')
```

## Mixture Properties

### Density

```python
# Mixture density
rho = PropsSI('D', 'P', P, T, mixture_spec)
```

### Enthalpy

```python
# Mixture enthalpy
h = PropsSI('H', 'P', P, T, mixture_spec)
```

### Entropy

```python
# Mixture entropy
s = PropsSI('S', 'P', P, T, mixture_spec)
```

### Internal Energy

```python
# Mixture internal energy
u = PropsSI('U', 'P', P, T, mixture_spec)
```

### Heat Capacity

```python
# Constant pressure heat capacity
cp = PropsSI('Cp', 'P', P, T, mixture_spec)

# Constant volume heat capacity
cv = PropsSI('Cv', 'P', P, T, mixture_spec)
```

## Phase Equilibrium

### Bubble Point

```python
# Bubble point temperature
T_bubble = PropsSI('T', 'P', P, 'Q', 0.0, mixture_spec)
```

### Dew Point

```python
# Dew point temperature
T_dew = PropsSI('T', 'P', P, 'Q', 1.0, mixture_spec)
```

### Phase Envelope

```python
# Get phase information
phase = PropsSI('Phase', 'P', P, 'Q', quality, mixture_spec)
# Returns: 'liquid', 'vapor', 'twophase', etc.
```

## Mixture Calculations

### Mole Fractions

```python
# Calculate mixture properties from mole fractions
# Component1: 0.5, Component2: 0.3, Component3: 0.2
rho = PropsSI('D', 'P', P, T, 'COMP1[0.5]&COMP2[0.3]&COMP3[0.2]')
```

### Mass Fractions

```python
# Calculate mixture properties from mass fractions
# Component1: 0.5 kg, Component2: 0.3 kg, Component3: 0.2 kg
rho = PropsSI('D', 'P', P, T, 'COMP1[0.5]&COMP2[0.3]&COMP3[0.2]')
```

### Partial Molar Properties

```python
# Partial molar enthalpy
h_molar = PropsSI('Hmolar', 'P', P, T, mixture_spec)

# Partial molar volume
V_molar = PropsSI('Vmolar', 'P', P, T, mixture_spec)
```

## Performance Considerations

### Mixture Complexity

| Mixture Type | Complexity | Calculation Speed |
|--------------|------------|-----------------|
| Binary | Low | Fastest |
| Ternary | Medium | Medium |
| Multi-component | High | Slower |

### Optimization Tips

1. **Use predefined mixtures**: Faster calculations
2. **Specify phase**: Avoid flash calculations
3. **Provide good initial guesses**: For complex mixtures
4. **Use appropriate backend**: IF97 for steam/water
5. **Batch calculations**: Group similar calculations

## Common Issues and Solutions

### Convergence Failures

**Problem**: Solver fails to find phase equilibrium

**Solutions**:
- Provide better initial guess
- Use phase imposition
- Check component specification format
- Try different equation of state

### Negative Properties

**Problem**: Properties return negative values

**Solutions**:
- Check input pair validity
- Verify mixture specification format
- Check temperature and pressure ranges
- Ensure phase specification is correct

### Phase Determination Errors

**Problem**: Cannot determine phase

**Solutions**:
- Use phase imposition: `T|liquid` or `T|gas`
- Check input pair defines valid state point
- Verify critical properties
- Check mixture specification format

## Advanced Topics

- **Excess Helmholtz energy**: Detailed energy terms
- **Fugacity calculations**: Non-ideal mixture behavior
- **Transport properties**: Mixture viscosity and conductivity
- **Speed of calculation**: Mixture sound speed
- **Critical properties**: Mixture critical points

## Best Practices

1. **Use predefined mixtures**: When available
2. **Verify composition**: Ensure sum of fractions = 1
3. **Check phase validity**: Verify state point is valid
4. **Use appropriate units**: SI units throughout
5. **Handle exceptions**: Catch calculation failures
6. **Validate results**: Check physical reasonableness
