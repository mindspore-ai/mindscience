# Incompressible Fluids

CoolProp provides incompressible fluid properties and correlations.

## Incompressible Fluids

### Water

**Characteristics**:
- Liquid water properties
- Valid from 273.16 K to 647.096 K
- Valid up to 100 MPa
- Transport property correlations available

**Properties available**:
- Density, enthalpy, entropy, internal energy
- Viscosity, conductivity, Prandtl number
- Speed of sound, compressibility factor
- Surface tension

**Example usage**:
```python
from CoolProp.CoolProp import PropsSI

# Density at 300 K, 10 MPa
rho = PropsSI('D', 'T', 300.0, 'P', 10e6, 'Water')

# Viscosity at 300 K, 10 MPa
mu = PropsSI('viscosity', 'T', 300.0, 'P', 10e6, 'Water')

# Specific heat
cp = PropsSI('Cp', 'T', 300.0, 'P', 10e6, 'Water')
```

### Refrigerants

**Characteristics**:
- Various refrigerant fluids available
- Valid over wide temperature ranges
- Transport property correlations

**Available refrigerants**:
- R134a (Refrigerant 134a)
- R125 (Propane)
- R22 (Difluoromethane)
- R1234yf (1,1,1,2-Tetrafluoroethane)
- R404A (R407C)
- R407C (R407C)
- R410A (R410A)

**Example usage**:
```python
# R134a properties
rho = PropsSI('D', 'T', 250.0, 'P', 101325.0, 'R134a')
cp = PropsSI('Cp', 'T', 250.0, 'P', 101325.0, 'R134a')
```

### Incompressible Liquid Mixtures

**Characteristics**:
- Multi-component liquid systems
- High-accuracy mixture properties
- Phase equilibrium calculations

**Example usage**:
```python
# Water-Ethylene glycol mixture
rho = PropsSI('D', 'P', 101325.0, 'T', 300.0, 
              'HEOS::EthyleneGlycol[0.3]&Water[0.7]')
```

## Partial Derivatives

CoolProp provides partial derivatives for incompressible fluids:

### First Partial Derivatives

```python
# Constant pressure specific heat
dH_dT = PropsSI('d(H)/d(T)|P', 'P', 101325.0, 'T', 300.0, 'Water')
```

### Second Partial Derivatives

```python
# Constant pressure heat capacity derivative
d2Cp_dT2 = PropsSI('d(Cp)/d(T)|P)/d(H)/d(T)|P', 
                 'P', 101325.0, 'T', 300.0, 'Water')
```

## Fitting Reports

CoolProp can generate fitting reports for property correlations:

### Property Fitting

```python
from CoolProp.CoolProp import CoolProp

CP = CoolProp.CoolProp()

# Generate fitting report
CP.write_fitting_report('Water', 'Cp', 'T', 280.0, 320.0)
```

### Report Contents

Fitting reports include:
- Correlation coefficients
- Statistical measures (R², AARD, MAXERR)
- Validity ranges
- Comparison with experimental data

## Transport Properties

CoolProp provides correlations for transport properties:

### Viscosity Correlations

```python
# Water viscosity at 300 K
mu = PropsSI('viscosity', 'T', 300.0, 'P', 101325.0, 'Water')
```

### Conductivity Correlations

```python
# Water thermal conductivity at 300 K
k = PropsSI('conductivity', 'T', 300.0, 'P', 101325.0, 'Water')
```

### Prandtl Number Correlations

```python
# Water Prandtl number at 300 K
Pr = PropsSI('Prandtl', 'T', 300.0, 'P', 101325.0, 'Water')
```

### Surface Tension Correlations

```python
# Water surface tension at 300 K
sigma = PropsSI('surface_tension', 'T', 300.0, 'P', 101325.0, 'Water')
```

## Fluid Selection Guide

| Application | Recommended Fluid | Reason |
|------------|------------------|---------|
| Water systems | Water | Most comprehensive data |
| Air conditioning | Air | Standard conditions |
| Refrigeration | R134a, R125 | High accuracy |
| Cryogenics | Nitrogen, CO2 | Low temperature |
| Combustion | CO2, Air | Fuel mixtures |
| Chemical processes | NH3, Ammonia | Chemical industry |

## Performance Considerations

### Input Pair Selection

| Input Pair | Speed | Use Case |
|-------------|-------|---------|
| ('T', 'P') | Fastest | Standard state |
| ('P', 'T') | Slower | P-T state |
| ('D', 'T') | Slower | D-T state |
| Other pairs | Slowest | Requires additional calculations |

### Backend Selection

| Backend | Speed | Use Case |
|---------|-------|---------|
| HEOS (default) | Fast | Most applications |
| IF97 | Fastest | Industrial steam/water |
| REFPROP | Fast | REFPROP available |
| BICUBIC | Slow | Tabular interpolation |

### Phase Imposition

For faster calculations in known phase regions:

```python
# Liquid phase (faster)
rho_liquid = PropsSI('D', 'T|liquid', 300.0, 'P', 101325.0, 'Water')

# Gas phase (faster)
rho_gas = PropsSI('D', 'T|gas', 300.0, 'P', 101325.0, 'Water')
```

## Common Issues and Solutions

### Invalid Input Parameters

**Problem**: Property calculation fails

**Solutions**:
- Check input pair validity (T, P in valid range)
- Verify fluid name spelling
- Check phase specification if imposed
- Ensure sufficient input parameters

### Convergence Failures

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

### Phase Determination Errors

**Problem**: Incorrect phase determined

**Solutions**:
- Check input pair defines valid state point
- Use phase imposition carefully
- Verify critical properties
- Check phase boundaries

## Advanced Topics

- **Extended corresponding states**: For transport properties
- **Critical point phenomena**: Near-critical behavior
- **Metastable properties**: Property extrapolation
- **Multiphase calculations**: High-pressure properties
- **Transport property models**: Viscosity, conductivity models

## Best Practices

1. **Use appropriate input pairs**: ('T', 'P') for fastest calculations
2. **Check validity ranges**: Verify inputs are within valid ranges
3. **Handle exceptions**: Catch and handle calculation failures
4. **Use phase imposition**: For known phase regions
5. **Set reference states**: For relative property calculations
6. **Verify results**: Check physical reasonableness
7. **Use appropriate backend**: HEOS, IF97, or REFPROP as needed
8. **Batch calculations**: Group similar calculations for efficiency
9. **Cache results**: For repeated calculations
10. **Validate inputs**: Check input parameters before calculations
