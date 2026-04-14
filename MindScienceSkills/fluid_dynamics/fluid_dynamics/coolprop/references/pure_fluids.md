# Pure and Pseudo-Pure Fluids

CoolProp provides high-accuracy thermodynamic properties for pure and pseudo-pure fluids.

## Pure Fluids

### Water

**Characteristics**:
- Most comprehensive fluid in CoolProp
- Valid from 273.16 K to 647.096 K
- Valid up to 22.064 MPa
- Equation of state: IAPWS-IF97
- Transport properties available

**Properties available**:
- Density, enthalpy, entropy, internal energy
- Viscosity, conductivity, Prandtl number
- Speed of sound, compressibility factor
- Surface tension, dielectric constant

**Example usage**:
```python
from CoolProp.CoolProp import PropsSI

# Density at 300 K, 1 atm
rho = PropsSI('D', 'T', 300.0, 'P', 101325.0, 'Water')

# Enthalpy at 300 K, 1 atm
h = PropsSI('H', 'T', 300.0, 'P', 101325.0, 'Water')

# Speed of sound at 300 K
c = PropsSI('speed_of_sound', 'T', 300.0, 'P', 101325.0, 'Water')
```

### Air

**Characteristics**:
- Valid from 59.15 K to 2000 K
- Valid up to 100 MPa
- Equation of state: Ideal gas with virial corrections
- Transport properties available

**Properties available**:
- Density, enthalpy, entropy, internal energy
- Viscosity, conductivity, Prandtl number
- Speed of sound, compressibility factor
- Virial coefficients

**Example usage**:
```python
# Density at 300 K, 1 atm
rho = PropsSI('D', 'T', 300.0, 'P', 101325.0, 'Air')

# Viscosity at 300 K, 1 atm
mu = PropsSI('viscosity', 'T', 300.0, 'P', 101325.0, 'Air')

# Prandtl number at 300 K
Pr = PropsSI('Prandtl', 'T', 300.0, 'P', 101325.0, 'Air')
```

### Nitrogen

**Characteristics**:
- Valid from 63.15 K to 7000 K
- Valid up to 100 MPa
- Equation of state: Ideal gas with virial corrections
- Cryogenic fluid properties available

**Properties available**:
- Density, enthalpy, entropy, internal energy
- Viscosity, conductivity, Prandtl number
- Speed of sound, compressibility factor
- Virial coefficients

**Example usage**:
```python
# Density at 298.15 K, 1 atm
rho = PropsSI('D', 'T', 298.15, 'P', 101325.0, 'Nitrogen')

# Critical temperature
T_crit = PropsSI('Tcrit', 'Nitrogen')

# Critical pressure
P_crit = PropsSI('Pcrit', 'Nitrogen')
```

### CO2

**Characteristics**:
- Valid from 216. K to 1100 K
- Valid up to 100 MPa
- Equation of state: Span-Wagner EOS
- Transport properties available

**Properties available**:
- Density, enthalpy, entropy, internal energy
- Viscosity, conductivity, Prandtl number
- Speed of sound, compressibility factor

**Example usage**:
```python
# Density at 300 K, 1 atm
rho = PropsSI('D', 'T', 300.0, 'P', 101325.0, 'CO2')

# Critical temperature
T_crit = PropsSI('Tcrit', 'CO2')

# Critical pressure
P_crit = PropsSI('Pcrit', 'CO2')
```

## Pseudo-Pure Fluids

### R134a (Refrigerant)

**Characteristics**:
- Valid from 169.48 K to 455.02 K
- Valid up to 6 MPa
- Equation of state: Helmholtz energy formulation
- Transport properties available

**Properties available**:
- Density, enthalpy, entropy, internal energy
- Viscosity, conductivity, Prandtl number
- Speed of sound, compressibility factor
- Surface tension

**Example usage**:
```python
# Density at 250 K, 1 atm
rho = PropsSI('D', 'T', 250.0, 'P', 101325.0, 'R134a')

# Enthalpy at 250 K, 1 atm
h = PropsSI('H', 'T', 250.0, 'P', 101325.0, 'R134a')

# Surface tension
sigma = PropsSI('surface_tension', 'T', 250.0, 'P', 101325.0, 'R134a')
```

### R125 (Propane)

**Characteristics**:
- Valid from 85.5 K to 369.83 K
- Valid up to 4.6 MPa
- Equation of state: Helmholtz energy formulation
- Transport properties available

**Properties available**:
- Density, enthalpy, entropy, internal energy
- Viscosity, conductivity, Prandtl number
- Speed of sound, compressibility factor
- Surface tension

**Example usage**:
```python
# Density at 300 K, 1 atm
rho = PropsSI('D', 'T', 300.0, 'P', 101325.0, 'R125')

# Critical temperature
T_crit = PropsSI('Tcrit', 'R125')

# Critical pressure
P_crit = PropsSI('Pcrit', 'R125')
```

### Ammonia (NH3)

**Characteristics**:
- Valid from 195.41 K to 405.4 K
- Valid up to 11.3 MPa
- Equation of state: Helmholtz energy formulation
- Transport properties available

**Properties available**:
- Density, enthalpy, entropy, internal energy
- Viscosity, conductivity, Prandtl number
- Speed of sound, compressibility factor
- Surface tension

**Example usage**:
```python
# Density at 300 K, 1 atm
rho = PropsSI('D', 'T', 300.0, 'P', 101325.0, 'NH3')

# Enthalpy at 300 K, 1 atm
h = PropsSI('H', 'T', 300.0, 'P', 101325.0, 'NH3')

# Surface tension
sigma = PropsSI('surface_tension', 'T', 300.0, 'P', 101325.0, 'NH3')
```

## Fluid Selection Guide

| Application | Recommended Fluid | Reason |
|------------|-----------------|---------|
| Water systems | Water | Most comprehensive data |
| Air conditioning | Air | Standard atmospheric conditions |
| Refrigeration | R134a, R125 | High accuracy properties |
| Cryogenics | Nitrogen, CO2 | Low temperature applications |
| Combustion | CO2, Air | Fuel mixtures |
| Chemical processes | NH3, Ammonia | Chemical industry |

## Property Calculation Patterns

### Single Property

```python
# Temperature-dependent density
for T in [280, 290, 300, 310, 320]:
    rho = PropsSI('D', 'T', T, 'P', 101325.0, 'Water')
    print(f"T={T} K, rho={rho} kg/m³")
```

### Multiple Properties

```python
# Calculate multiple properties at once
properties = ['D', 'H', 'S', 'Cp', 'viscosity']
T, P = 300.0, 101325.0

for prop in properties:
    value = PropsSI(prop, 'T', T, 'P', P, 'Water')
    print(f"{prop} = {value}")
```

### Property Ranges

```python
# Check validity range
T_min = PropsSI('Tmin', 'Water')
T_max = PropsSI('Tmax', 'Water')

# Critical properties
T_crit = PropsSI('Tcrit', 'Water')
P_crit = PropsSI('Pcrit', 'Water')
```

## Performance Considerations

### Input Pair Selection

CoolProp uses different calculation speeds based on input pairs:

| Input Pair | Speed | Use Case |
|-------------|-------|---------|
| ('T', 'P') | Fastest | Most common, T-ρ state variables |
| ('P', 'T') | Slower | P-T state variables |
| ('D', 'T') | Slower | D-T state variables |
| Other pairs | Slowest | Requires additional calculations |

### Phase Imposition

For faster calculations in known phase regions:

```python
# Liquid phase (faster)
rho_liquid = PropsSI('D', 'T|liquid', 300.0, 'P', 101325.0, 'Water')

# Gas phase (faster)
rho_gas = PropsSI('D', 'T|gas', 300.0, 'P', 101325.0, 'Water')
```

### Reference States

For relative property calculations (enthalpy, entropy):

```python
from CoolProp.CoolProp import set_reference_stateS

# Set reference state for enthalpy
set_reference_stateS('IIR', 'Water')  # 200 kJ/kg reference
```

## Common Issues and Solutions

### Invalid Input Parameters

**Problem**: Property calculation fails

**Solutions**:
- Check input pair validity (T, P in valid range)
- Verify fluid name spelling
- Check phase specification if imposed
- Ensure sufficient input parameters

### Phase Determination Failures

**Problem**: Cannot determine phase

**Solutions**:
- Provide phase specification: `T|liquid`
- Check input pair defines valid state point
- Use phase imposition carefully
- Verify critical properties

### Convergence Issues

**Problem**: Solver fails to converge

**Solutions**:
- Provide better initial guess
- Use phase imposition
- Check input pair consistency
- Try different equation of state

### Accuracy Concerns

**Problem**: Results seem inaccurate

**Solutions**:
- Use high-accuracy backend (IF97)
- Check reference state settings
- Verify input pair validity
- Compare with experimental data

## Best Practices

1. **Use appropriate input pairs**: ('T', 'P') for fastest calculations
2. **Check validity ranges**: Verify inputs are within valid ranges
3. **Handle exceptions**: Catch and handle calculation failures
4. **Use phase imposition**: For known phase regions
5. **Set reference states**: For relative property calculations
6. **Batch calculations**: Group similar calculations for efficiency
7. **Cache results**: For repeated calculations
8. **Verify results**: Check physical reasonableness
