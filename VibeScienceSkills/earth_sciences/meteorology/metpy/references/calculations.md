# Calculations

MetPy provides comprehensive calculation capabilities for meteorological data.

## Thermodynamic Calculations

### Potential Temperature

```python
from metpy.calc import thermo

# Calculate potential temperature
theta = thermo.potential_temperature(temperature, pressure)

# Calculate equivalent potential temperature
theta_e = thermo.equivalent_potential_temperature(temperature, pressure)
```

### Virtual Temperature

```python
from metpy.calc import thermo

# Calculate virtual temperature
tv = thermo.virtual_temperature(temperature, pressure, mixing_ratio)
```

### Equivalent Potential Temperature

```python
from metpy.calc import thermo

# Calculate equivalent potential temperature
theta_e = thermo.equivalent_potential_temperature(temperature, pressure)
```

### Moisture Calculations

```python
from metpy.calc import thermo

# Calculate mixing ratio
w = thermo.mixing_ratio(temperature, dewpoint)

# Calculate relative humidity
rh = thermo.relative_humidity(temperature, dewpoint)

# Calculate dewpoint temperature
td = thermo.dewpoint(temperature, relative_humidity)
```

## Kinematic Calculations

### Wind Speed and Direction

```python
from metpy.calc import kinematics

# Calculate wind speed
speed = kinematics.wind_speed(u, v)

# Calculate wind direction
direction = kinematics.wind_direction(u, v)
```

### Wind Components

```python
from metpy.calc import kinematics

# Calculate u and v components from speed and direction
u, v = kinematics.wind_components(speed, direction)
```

### Wind Shear

```python
from metpy.calc import kinematics

# Calculate wind shear
shear = kinematics.wind_shear(u, v, level1, level2)
```

## Dynamic Calculations

### Vorticity

```python
from metpy.calc import dynamics

# Calculate relative vorticity
vort_rel = dynamics.relative_vorticity(u, v)

# Calculate absolute vorticity
vort_abs = dynamics.absolute_vorticity(u, v)
```

### Divergence

```python
from metpy.calc import dynamics

# Calculate divergence
div = dynamics.divergence(u, v)
```

### Advection

```python
from metpy.calc import advection

# Advect scalar field
advected = advection.advection(u, v, scalar_field)
```

### Deformation

```python
from metpy.calc import dynamics

# Calculate deformation
deformation = dynamics.deformation(u, v)
```

## Stability Calculations

### Richardson Number

```python
from metpy.calc import stability

# Calculate Richardson number
ri = stability.richardson_number(temperature, wind_shear, potential_temperature_gradient)
```

### Brunt-Vaisala Frequency

```python
from metpy.calc import stability

# Calculate Brunt-Vaisala frequency
n2 = stability.brunt_vaisala_squared(temperature, pressure)
```

### Static Stability

```python
from metpy.calc import stability

# Calculate static stability
stability = stability.static_stability(temperature, pressure)
```

## Precipitation Calculations

### Precipitable Water

```python
from metpy.calc import precip

# Calculate precipitable water
pwat = precip.precipitable_water(specific_humidity, pressure)
```

### Precipitation Rate

```python
from metpy.calc import precip

# Calculate precipitation rate
rate = precip.precipitation_rate(precipitable_water, time_step)
```

## Common Issues and Solutions

### Calculation Failures

**Problem**: Calculation fails

**问题**: 计算失败

**Solutions**:
- Check input data validity
- Verify data dimensions
- Check for missing values
- Verify coordinate systems

### Unit Issues

**Problem**: Results in wrong units

**问题**: 结果单位错误

**Solutions**:
- Check input units
- Verify output units
- Convert to appropriate units
- Check unit consistency

### Stability Issues

**Problem**: Calculations become unstable

**问题**: 计算变得不稳定

**Solutions**:
- Check data quality
- Verify coordinate spacing
- Check for singularities
- Apply smoothing if needed

## Best Practices

1. **Validate input data**: Check data quality before calculations
2. **Check units**: Ensure consistent units
3. **Handle missing values**: Account for missing data
4. **Document calculations**: Keep track of calculation parameters
5. **Verify results**: Check physical reasonableness
6. **Use appropriate methods**: Match method to data type
