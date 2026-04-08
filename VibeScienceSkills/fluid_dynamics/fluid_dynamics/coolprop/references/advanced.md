# Advanced CoolProp Features

CoolProp provides advanced features for specialized applications.

## Extended Corresponding States

CoolProp can use extended corresponding states for transport properties:

### What are Extended Corresponding States?

Standard corresponding states use enthalpy and entropy differences. Extended states use other properties like:

- Internal energy differences
- Helmholtz energy differences
- Gibbs energy differences
- Specific volume differences

### When to Use Extended States

**Use case**: When calculating transport properties where enthalpy/entropy differences are not sufficient

**Examples**:
```python
# Use extended state for internal energy
dU_dT = PropsSI('d(U)/d(T)|P', 'T', 300.0, 'P', 101325.0, 'Water')
```

### Available Extended States

| Extended State | Description | Use Case |
|--------------|-------------|---------|
| `d(U)/d(T)` | Internal energy difference | Transport properties |
| `d(H)/d(T)` | Helmholtz energy difference | Transport properties |
| `d(G)/d(T)` | Gibbs energy difference | Thermodynamic analysis |
| `d(V)/d(T)` | Specific volume difference | Compressibility calculations |

## Conversion from Ideal Gas Term to Helmholtz Energy Term

CoolProp provides conversion between ideal gas and Helmholtz energy formulations:

### Conversion Formula

```python
# Convert ideal gas term to Helmholtz energy term
h_helmholtz = PropsSI('d(H)/d(T)|P', 'T', 300.0, 'P', 101325.0, 'Water')
```

### When to Use This Conversion

**Use case**: When working with Helmholtz energy correlations that require Helmholtz energy terms

**Examples**:
```python
# Calculate Helmholtz energy from ideal gas term
h = PropsSI('H', 'T', 300.0, 'P', 101325.0, 'Water')
dH_dT = PropsSI('d(H)/d(T)|P', 'T', 300.0, 'P', 101325.0, 'Water')

# Convert to Helmholtz energy term
h_helmholtz = h - dH_dT
```

## Converting Bender and mBWR EOS

CoolProp can convert between Bender and mBWR equations of state:

### Bender to mBWR

```python
# Convert Bender EOS parameters to mBWR
# Requires specific parameters for conversion
```

### When to Use This Conversion

**Use case**: When using mBWR correlations with Bender parameters

**Examples**:
```python
# Bender parameters
T_reduced = 300.0  # Reduced temperature
P_reduced = 101325.0  # Reduced pressure

# Convert to mBWR parameters
# CoolProp handles the conversion internally
```

## Cubic Equations of State (SRK, PR)

CoolProp supports cubic equations of state for higher accuracy:

### SRK (Soave-Redlich-Kwong)

```python
# Use SRK cubic equation of state
# Higher accuracy than standard correlations
```

### PR (Peng-Robinson)

```python
# Use PR cubic equation of state
# Good balance of accuracy and computational cost
```

### When to Use Cubic Equations

**Use case**: When higher accuracy is required for property calculations

**Examples**:
```python
# Select cubic equation of state
# CoolProp automatically uses appropriate equation
```

## PC-SAFT Equation of State

CoolProp implements the PC-SAFT equation of state for water:

### What is PC-SAFT?

**Characteristics**:
- Polynomial form of equation of state
- Valid for wide temperature and pressure ranges
- Higher accuracy than IAPWS-IF97
- Fast evaluation

### When to Use PC-SAFT

**Use case**: When high accuracy water properties are needed

**Examples**:
```python
# PC-SAFT is used automatically for water
# No special configuration needed
```

### PC-SAFT Range of Validity

**Valid range**: 273.16 K to 647.096 K

**Use case**: Standard water property calculations

## Tabular Interpolation

CoolProp provides computationally efficient tabular interpolation:

### What is Tabular Interpolation?

**Characteristics**:
- Fast lookup from precomputed tables
- Good for repeated calculations
- Lower accuracy than equation of state
- Efficient for batch calculations

### When to Use Tabular Interpolation

**Use case**: When calculation speed is more important than accuracy

**Examples**:
```python
# Use tabular interpolation for speed
# CoolProp automatically selects appropriate method
```

### TTSE Interpolation

**Characteristics**:
- Trilinear table lookup
- Very fast
- Good for regular grids
- Limited accuracy

### Bicubic Interpolation

**Characteristics**:
- Bicubic interpolation
- Better accuracy than TTSE
- Slightly slower
- Good for smooth property variations

### Interpolation Selection Guide

| Requirement | Recommended Method | Reason |
|--------------|---------------------|---------|
| Maximum speed | TTSE | Fastest lookup |
| Good accuracy | Bicubic | Smooth interpolation |
| High accuracy | Equation of state | Most accurate |
| Batch calculations | Tabular | Efficient for many points |

## Configuration Variables

CoolProp provides configuration variables for customization:

### Available Configuration Variables

| Variable | Description | Use Case |
|-----------|-------------|---------|
| `OVERWRITE_FLUIDS` | Allow overwriting fluid definitions | Custom fluid definitions |
| `NORMALIZE_HYDRATE` | Normalize hydrate names | Consistent naming |
| `REFPROP_BINARY_PATH` | Path to REFPROP library | REFPROP integration |
| `ERR_HALT_IPS` | Halt on errors | Error handling |

### Setting Configuration Variables

```python
from CoolProp.CoolProp import CoolProp

CP = CoolProp.CoolProp()

# Set configuration variable
CP.set_config_bool(CP.OVERWRITE_FLUIDS, True)

# Check configuration value
value = CP.get_config_bool(CP.OVERWRITE_FLUIDS)
```

### Overwriting Fluid Definitions

```python
from CoolProp.CoolProp import CoolProp

CP = CoolProp.CoolProp()

# Enable overwriting
CP.set_config_bool(CP.OVERWRITE_FLUIDS, True)

# Add custom fluid
CP.add_fluids_as_JSON("HEOS", fluid_json_data)

# Disable overwriting
CP.set_config_bool(CP.OVERWRITE_FLUIDS, False)
```

## Performance Optimization

### Calculation Speed

| Optimization | Speed Improvement | Use Case |
|-------------|-------------------|---------|
| Input pair selection | 3-10x faster | Use ('T', 'P') |
| Phase imposition | 2-5x faster | For known phases |
| Tabular interpolation | 10-100x faster | For batch calculations |
| Vectorized calculations | 2-5x faster | For multiple properties |

### Memory Optimization

| Optimization | Memory Reduction | Use Case |
|-------------|------------------|---------|
| Reuse instances | 50-90% reduction | Single CP instance |
| Batch calculations | 70-80% reduction | Group similar calculations |
| Cache results | Variable | For repeated calculations |

### Accuracy vs Speed Trade-offs

| Accuracy Level | Calculation Method | Speed | Use Case |
|--------------|-------------------|-------|---------|
| Highest | Equation of state | Slowest | Critical calculations |
| High | Cubic equation | Slow | Accurate results |
| Medium | Tabular interpolation | Fast | Most applications |
| Low | TTSE interpolation | Fastest | Quick estimates |

## Common Issues and Solutions

### Solver Convergence Failures

**Problem**: Property solver fails to converge

**Solutions**:
- Provide better initial guess
- Use phase imposition
- Check input pair validity
- Try different equation of state

### Invalid Fluid Definitions

**Problem**: Custom fluid definition not valid

**Solutions**:
- Check JSON format
- Verify required fields
- Check property ranges
- Enable OVERWRITE_FLUIDS

### REFPROP Integration Issues

**Problem**: REFPROP library not found or incompatible

**Solutions**:
- Check REFPROP_BINARY_PATH
- Verify REFPROP installation
- Check library compatibility
- Use CoolProp backend instead

### Tabular Interpolation Errors

**Problem**: Interpolation outside valid range

**Solutions**:
- Check input values are within valid range
- Use equation of state for extended range
- Implement custom interpolation
- Add safety checks

## Advanced Topics

- **Multi-fluid calculations**: Handling multiple fluids simultaneously
- **Property derivatives**: First and second partial derivatives
- **Critical point phenomena**: Near-critical behavior
- **Metastable properties**: Phase change properties
- **Transport property models**: Custom viscosity/conductivity models
- **High-pressure properties**: Supercritical fluid behavior
- **Electrolyte properties**: Electrolyte solution properties
- **Aqueous properties**: Solution chemistry properties

## Best Practices

1. **Use appropriate input pairs**: ('T', 'P') for fastest calculations
2. **Check validity ranges**: Verify inputs are within valid ranges
3. **Handle exceptions**: Catch and handle calculation failures
4. **Use phase imposition**: For known phase regions
5. **Set reference states**: For relative property calculations
6. **Batch calculations**: Group similar calculations for efficiency
7. **Cache results**: For repeated calculations
8. **Choose appropriate backend**: HEOS, IF97, or REFPROP
9. **Validate results**: Check physical reasonableness
10. **Use configuration variables**: Customize behavior as needed
