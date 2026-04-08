# Interpolation in WRF-Python

Complete guide to interpolation routines and mapping between levels.

## Vertical Interpolation

### Pressure to Model Levels

```python
from wrf import getvar, vertcross

# Get temperature on pressure levels
t_p = getvar(ncfile, 'T', True)  # True = unpivoted

# Get geopotential height
h = getvar(ncfile, 'GHT', True)

# Interpolate to model levels
t_ml = vertcross(t_p, h)
print(f"Temperature on model levels: {t_ml.shape}")
```

### Model to Pressure Levels

```python
from wrf import pres

# Get pressure level data from model levels
# See WRF-Python documentation for details
```

## Geopotential Height Interpolation

### Get Geopotential Height

```python
from wrf import ght

# Get geopotential height (m)
ght = ght(ncfile, True)
print(f"Geopotential height shape: {ght.shape}")
```

### Terrain Height

```python
from wrf import getvar

# Get terrain height (m)
terrain = getvar(ncfile, 'Z', False)  # False = mass points
print(f"Terrain height shape: {terrain.shape}")
```

## Horizontal Interpolation

### Lat-Lon to Model Grid

```python
from wrf import geobucket

# Interpolate from lat-lon to model grid
# See WRF-Python documentation for details
```

### Model Grid to Lat-Lon

```python
# Interpolate from model grid to lat-lon
# See WRF-Python documentation for details
```

## Common Applications

### Vertical Cross Section

```python
from wrf import getvar, vertcross

# Get temperature on pressure levels
t_p = getvar(ncfile, 'T', True)

# Get geopotential height
h = getvar(ncfile, 'GHT', True)

# Interpolate to model levels
t_ml = vertcross(t_p, h)

# Extract cross section at specific location
lat_idx = 50  # Latitude index
lon_idx = 50  # Longitude index

cross_section = t_ml[:, :, lat_idx, lon_idx]
print(f"Cross section shape: {cross_section.shape}")
```

### Pressure Level Cross Section

```python
# Extract cross section on pressure level
level_idx = 10  # Pressure level index

cross_section = t_p[:, level_idx, :, :]
print(f"Pressure level cross section shape: {cross_section.shape}")
```

### Time Series at Point

```python
# Get time series at specific location
lat_idx = 50
lon_idx = 50
level_idx = 10

time_series = t_ml[:, level_idx, lat_idx, lon_idx]
print(f"Time series shape: {time_series.shape}")
```

### Time Series Averaged

```python
import numpy as np

# Average over spatial region
lat_slice = slice(40, 60)
lon_slice = slice(40, 60)

t_avg = t_ml[:, :, lat_slice, lon_slice].mean(axis=(2, 3))
print(f"Averaged time series shape: {t_avg.shape}")
```

## Mapping Between Levels

### Pressure to Model Mapping

```python
from wrf import getvar, vertcross

# Create mapping between pressure and model levels
t_p = getvar(ncfile, 'T', True)
h = getvar(ncfile, 'GHT', True)
t_ml = vertcross(t_p, h)

# The mapping is implicit in the interpolation
# Use the interpolated data for analysis
```

### Terrain Following

```python
# Interpolate following terrain
# See WRF-Python documentation for details
```

## Numerical Considerations

### Interpolation Accuracy

**Guidelines:**
- Use appropriate interpolation method
- Check for extrapolation
- Verify interpolation results
- Handle missing values

### Coordinate Systems

**WRF coordinate systems:**
- Mass points (staggered grid)
- Model levels (geopotential height)
- Pressure levels
- Be aware of grid staggering

### Domain Boundaries

**Handling boundaries:**
- Check for cyclic boundaries
- Handle missing values at boundaries
- Verify interpolation near boundaries
- Use appropriate boundary conditions

## Troubleshooting

### Interpolation Errors

**Issue:** Interpolation fails

**Solutions:**
- Check input array shapes
- Verify geopotential height availability
- Check for missing values
- Ensure coordinate compatibility

### Incorrect Results

**Issue:** Interpolated values are incorrect

**Solutions:**
- Verify interpolation method
- Check coordinate systems
- Validate input data
- Review WRF model configuration

### Memory Issues

**Issue:** Out of memory during interpolation

**Solutions:**
- Process in smaller chunks
- Use xarray for large files
- Reduce spatial resolution
- Close files after reading

### Coordinate Mismatches

**Issue:** Coordinate systems don't match

**Solutions:**
- Check grid staggering
- Verify coordinate dimensions
- Review WRF model configuration
- Use appropriate interpolation method

## Advanced Topics

### Custom Interpolation

```python
# Define custom interpolation method
# See WRF-Python documentation for details
```

### Multi-Variable Interpolation

```python
# Interpolate multiple variables together
# Ensure consistent coordinate systems
```

### Time-Dependent Interpolation

```python
# Interpolate at each time step
# Track interpolation over time
```

## Resources

- WRF-Python interpolation: https://wrf-python.readthedocs.io/en/latest/basic%5usage.html#interpolation-routines
- WRF-Python geobucket: https://wrf-python.readthedocs.io/en/latest/basic%5usage.html#mapping-helper-routines
- WRF-Python vertcross: https://wrf-python.readthedocs.io/en/latest/basic%5usage.html#lat-lon-xy-routines
