# WRF Coordinate Systems

Complete guide to WRF coordinate systems and grids.

## Coordinate Systems

### Mass Grid (A-Grid)

**Characteristics:**
- Staggered relative to velocity grid
- Temperature, humidity, pressure, etc.
- Cell centers
- Latitude-longitude coordinates

**Accessing:**
```python
from wrf import getvar

# Get temperature on mass grid
t = getvar(ncfile, 'T', False)  # False = mass grid
print(f"Temperature shape: {t.shape}")  # (time, level, lat, lon)
```

**Coordinates:**
```python
# Get coordinates
lat = getvar(ncfile, 'XLAT', False)
lon = getvar(ncfile, 'XLON', False)

print(f"Latitude range: [{lat.min():.2f}, {lat.max():.2f}]°")
print(f"Longitude range: [{lon.min():.2f}, {lon.max():.2f}]°")
```

### Velocity Grid (C-Grid)

**Characteristics:**
- Staggered relative to mass grid
- U, V, W wind components
- Cell edges
- Half-index shift

**Accessing:**
```python
# Get U wind component
u = getvar(ncfile, 'U', False)  # False = mass grid (shifted)
print(f"U wind shape: {u.shape}")
```

**Coordinate shift:**
```python
# Velocity grid is shifted by half index
# U is at i+1/2, j+1/2 relative to mass grid
```

### Terrain Grid (H-Grid)

**Characteristics:**
- Terrain height
- Mass grid location
- Constant in time

**Accessing:**
```python
# Get terrain height
terrain = getvar(ncfile, 'HGT', False)  # False = mass grid
print(f"Terrain shape: {terrain.shape}")
```

### Pressure Levels

**Characteristics:**
- Hybrid sigma-pressure coordinates
- Vertical coordinate
- Model levels and pressure levels

**Accessing:**
```python
# Get geopotential height
ght = getvar(ncfile, 'GHT', False)  # False = mass grid
print(f"Geopotential height shape: {ght.shape}")
```

## Grid Staggering

### Mass-Velocity Staggering

```python
# Mass grid (cell centers)
t = getvar(ncfile, 'T', False)

# Velocity grid (cell edges)
u = getvar(ncfile, 'U', False)

# Velocity grid is shifted by half index
# U[i,j] is at U[i+1/2, j+1/2] relative to T[i,j]
```

### Pressure-Mass Staggering

```python
# Mass grid (sigma levels)
t_p = getvar(ncfile, 'T', True)  # True = pressure levels

# Geopotential height
ght = getvar(ncfile, 'GHT', True)

# Model levels (mass grid location)
# See interpolation for mapping
```

## Coordinate Transformations

### Lat-Lon to Grid Indices

```python
# Find nearest grid point
import numpy as np

lat_point = 40.0
lon_point = -100.0

# Find indices
lat_idx = np.argmin(np.abs(lat[0, 0, :] - lat_point))
lon_idx = np.argmin(np.abs(lon[0, 0, :] - lon_point))

print(f"Nearest grid point: ({lat_idx}, {lon_idx})")
print(f"Coordinates: ({lat[0, 0, lat_idx, lon_idx]:.2f}°, {lon[0, 0, lat_idx, lon_idx]:.2f}°)")
```

### Grid Indices to Lat-Lon

```python
# Get coordinates at grid point
lat_idx = 50
lon_idx = 50

lat_coord = lat[0, 0, lat_idx, lon_idx]
lon_coord = lon[0, 0, lat_idx, lon_idx]

print(f"Coordinates: ({lat_coord:.2f}°, {lon_coord:.2f}°)")
```

## Domain Boundaries

### Cyclic Boundaries

```python
# WRF uses cyclic boundaries in x-direction
# Handle boundary conditions appropriately
```

### Domain Extent

```python
# Check domain extent
lat_min, lat_max = lat.min(), lat.max()
lon_min, lon_max = lon.min(), lon.max()

print(f"Domain extent:")
print(f"  Latitude: [{lat_min:.2f}, {lat_max:.2f}]°")
print(f"  Longitude: [{lon_min:.2f}, {lon_max:.2f}]°")
```

### Grid Spacing

```python
# Calculate grid spacing
dlat = np.mean(np.diff(lat[0, 0, :, 0]))
dlon = np.mean(np.diff(lon[0, 0, 0, :]))

print(f"Grid spacing:")
print(f"  dlat: {dlat:.4f}°")
print(f"  dlon: {dlon:.4f}°")
```

## Common Applications

### Domain Analysis

```python
# Analyze domain characteristics
lat = getvar(ncfile, 'XLAT', False)
lon = getvar(ncfile, 'XLON', False)
terrain = getvar(ncfile, 'HGT', False)

print(f"Domain size: {lat.size} x {lon.size}")
print(f"Terrain range: [{terrain.min():.1f}, {terrain.max():.1f}] m")
```

### Grid Resolution

```python
# Check grid resolution
dlat = np.mean(np.diff(lat[0, 0, :, 0]))
dlon = np.mean(np.diff(lon[0, 0, 0, :]))

print(f"Grid resolution: {dlat:.4f}° x {dlon:.4f}°")
```

### Coordinate Validation

```python
# Validate coordinate ranges
lat = getvar(ncfile, 'XLAT', False)
lon = getvar(ncfile, 'XLON', False)

assert -90 <= lat.min() and lat.max() <= 90, "Latitude out of range"
assert -180 <= lon.min() and lon.max() <= 180, "Longitude out of range"

print("Coordinates validated")
```

## Numerical Considerations

### Grid Quality

**Guidelines:**
- Check grid spacing uniformity
- Verify coordinate ranges
- Validate staggering
- Check for missing values

### Boundary Handling

**Cyclic boundaries:**
- Handle x-direction cyclic boundaries
- Be aware of boundary conditions
- Check for edge effects
- Validate interpolation near boundaries

### Staggering Effects

**Mass-velocity staggering:**
- Velocity grid is shifted by half index
- Interpolation may be needed
- Check for phase shifts
- Verify coordinate alignment

## Troubleshooting

### Coordinate Errors

**Issue:** Incorrect coordinate ranges

**Solutions:**
- Check WRF model configuration
- Verify domain settings
- Validate grid staggering
- Review WRF documentation

### Staggering Errors

**Issue:** Incorrect staggering between grids

**Solutions:**
- Verify grid types (mass vs velocity)
- Check coordinate shifts
- Review interpolation methods
- Consult WRF documentation

### Missing Coordinates

**Issue:** Coordinate variables not found

**Solutions:**
- Check variable names
- Verify netCDF file structure
- Review WRF model configuration
- Check for different coordinate systems

## Advanced Topics

### Nested Grids

```python
# WRF supports nested grids
# See WRF documentation for details
```

### Moving Grids

```python
# Moving nest grids
# See WRF documentation for details
```

### Grid Rotation

```python
# Rotate coordinates
# See WRF documentation for details
```

## Resources

- WRF-ARW model: https://www2.mmm.ucar.edu/wrf/users/
- WRF coordinate systems: https://www2.mmm.ucar.edu/wrf/users/src/
- WRF-Python interpolation: https://wrf-python.readthedocs.io/en/latest/basic%5usage.html#interpolation-routines
