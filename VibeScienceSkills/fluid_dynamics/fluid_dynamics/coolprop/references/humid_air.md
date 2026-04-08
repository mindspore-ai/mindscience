# Humid Air Properties

CoolProp provides high-accuracy psychrometric calculations for humid air.

## Humid Air Basics

Humid air is a mixture of dry air and water vapor. CoolProp provides accurate calculations for:

- Relative humidity and dew point
- Wet bulb and dry bulb temperatures
- Enhancement factor for moist air
- Isothermal and isobaric properties

## Basic Usage

### HumidAir Class

```python
from CoolProp.CoolProp import HumidAir

# Create humid air instance
ha = HumidAir(T=298.15, P=101325.0, RH=0.5)

# Get properties
W_bulb = ha.W_bulb()      # Wet bulb temperature (K)
T_dp = ha.T_dp()          # Dew point temperature (K)
enhancement = ha.enhancement_factor()  # Enhancement factor
```

### Properties Available

**Temperature properties**:
- `W_bulb`: Wet bulb temperature
- `T_dp`: Dew point temperature
- `T_dp_WB`: Dew point wet bulb temperature

**Humidity properties**:
- `RH`: Relative humidity (0-1)
- `omega`: Humidity ratio (0-1)
- `p_v`: Vapor pressure (Pa)

**Derived properties**:
- `enhancement_factor`: Enhancement factor for moist air
- `isothermal_compressibility`: Isothermal compressibility factor
- `isobaric_expansion_coefficient`: Isobaric expansion coefficient

## Humidity Calculations

### Relative Humidity

```python
from CoolProp.CoolProp import HumidAir

# Calculate RH from T, P, and T_wb
haT = HumidAir(T=298.15, P=101325.0, W_bulb=280.0)
rh = haT.RH  # Relative humidity
```

### Humidity Ratio

```python
# Calculate humidity ratio
ha = HumidAir(T=298.15, P=101325.0, RH=0.5)
omega = ha.omega  # Humidity ratio (kg_water/kg_air)
```

### Vapor Pressure

```python
# Calculate vapor pressure
ha = HumidAir(T=298.15, P=101325.0, RH=0.5)
p_v = ha.p_v  # Vapor pressure (Pa)
```

## Temperature Calculations

### Wet Bulb Temperature

```python
# Calculate wet bulb temperature
ha = HumidAir(T=298.15, P=101325.0, RH=0.5)
W_bulb = ha.W_bulb  # Wet bulb temperature (K)
```

### Dew Point Temperature

```python
# Calculate dew point temperature
ha = HumidAir(T=298.15, P=101325.0, RH=0.5)
T_dp = ha.T_dp  # Dew point temperature (K)
```

### Dew Point Wet Bulb Temperature

```python
# Calculate dew point wet bulb temperature
ha = HumidAir(T=298.15, P=101325.0, RH=0.5)
T_dp_WB = ha.T_dp_WB  # Dew point wet bulb temperature (K)
```

## Enhancement Factor

The enhancement factor accounts for the effect of moisture on air properties:

```python
# Get enhancement factor
ha = HumidAir(T=298.15, P=101325.0, RH=0.5)
enhancement = ha.enhancement_factor()

# Calculate enhanced properties
rho_dry = PropsSI('D', 'T', T, 'P', P, 'Air')
rho_moist = rho_dry * enhancement
```

## Compressibility and Expansion

### Isothermal Compressibility

```python
# Isothermal compressibility factor
ha = HumidAir(T=298.15, P=101325.0, RH=0.5)
kappa_T = ha.isothermal_compressibility  # Isothermal compressibility factor
```

### Isobaric Expansion

```python
# Isobaric expansion coefficient
ha = HumidAir(T=298.15, P=101325.0, RH=0.5)
mu_P = ha.isobaric_expansion_coefficient  # Isobaric expansion coefficient
```

## Psychrometric Charts

### Mollier Diagram

```python
# Create humid air instance
ha = HumidAir(T=298.15, P=101325.0, RH=0.5)

# Get psychrometric chart data
# CoolProp provides methods to access chart data
# Use for plotting and analysis
```

### Temperature-Relative Humidity

```python
# Calculate T-RH relationship
for RH in [0.2, 0.4, 0.6, 0.8, 1.0]:
    ha = HumidAir(T=298.15, P=101325.0, RH=RH)
    W_bulb = ha.W_bulb
    print(f"RH={RH:.2f}, W_bulb={W_bulb:.2f} K")
```

## Validation

### Humid Air Validation

CoolProp includes validation for humid air properties:

```python
# Validation ensures physical consistency
# Check for invalid input combinations
# Verify property ranges and relationships
```

### Verification Script

```python
from CoolProp.CoolProp import HumidAir

# Test various conditions
test_cases = [
    (298.15, 101325.0, 0.5),  # Standard conditions
    (280.0, 101325.0, 0.8),   # High humidity
    (320.0, 101325.0, 0.3),   # Low humidity
]

for T, P, RH in test_cases:
    ha = HumidAir(T, P, RH)
    print(f"T={T} K, P={P} Pa, RH={RH}")
    print(f"W_bulb={ha.W_bulb} K, T_dp={ha.T_dp} K")
```

## Performance Considerations

### Calculation Speed

HumidAir property calculations are very fast:
- Temperature properties: Instant
- Humidity properties: Instant
- Derived properties: Fast

### Memory Usage

HumidAir instances are lightweight:
- Single instance per state point
- No large memory overhead
- Efficient for batch calculations

### Accuracy

CoolProp provides high accuracy:
- Psychrometric calculations validated against standards
- Enhancement factor from high-accuracy correlations
- Consistent with other CoolProp properties

## Common Issues and Solutions

### Condensation Problems

**Problem**: Condensation occurs unexpectedly

**Solutions**:
- Check dew point temperature
- Verify relative humidity
- Consider surface effects
- Check thermal insulation

### Humidity Sensor Errors

**Problem**: Sensor readings seem incorrect

**Solutions**:
- Verify sensor calibration
- Check temperature sensor accuracy
- Validate RH measurement range
- Consider sensor response time

### Enhancement Factor Issues

**Problem**: Enhancement factor seems incorrect

**Solutions**:
- Verify humidity calculation
- Check temperature and pressure accuracy
- Validate against experimental data
- Consider local variations

## Advanced Topics

- **Adiabatic processes**: Humidity variations
- **HVAC applications**: System design calculations
- **Meteorology**: Atmospheric humidity calculations
- **Industrial drying**: Drying process optimization
- **Food storage**: Product quality control

## Best Practices

1. **Use HumidAir class**: For humid air properties
2. **Check validity ranges**: Verify inputs are reasonable
3. **Handle condensation**: Consider dew point in designs
4. **Use enhancement factor**: For accurate moist air properties
5. **Validate results**: Check against experimental data
6. **Consider pressure effects**: Atmospheric pressure variations
7. **Batch calculations**: Process multiple state points efficiently
8. **Error handling**: Catch and handle calculation failures
