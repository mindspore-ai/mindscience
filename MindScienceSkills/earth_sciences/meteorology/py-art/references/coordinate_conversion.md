# Coordinate Conversion with Py-ART

This guide covers polar to Cartesian coordinate conversion methods.

## Overview

Radar data is typically in polar coordinates (azimuth, elevation, range). Many applications require conversion to Cartesian coordinates (x, y, z).

## Basic Conversion

### Constant Azimuth and Range

Convert to Cartesian grid with constant azimuth and range:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Convert to Cartesian grid
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity', 'velocity'],
    edge_factor=0.0
)

# Display grid information
print(grid)
print(f"Grid shape: {grid.fields['reflectivity']['data'].shape}")
```

### Grid from Radars

Convert multiple radars to common grid:

```python
import pyart

# Read multiple radars
radar1 = pyart.io.read_arm_netcdf('radar1.nc')
radar2 = pyart.io.read_arm_netcdf('radar2.nc')

# Convert to common grid
grid = pyart.map.grid_from_radars(
    (radar1, radar2), 
    grid_shape=(500, 500),
    grid_limits=((0, 50000), (0, 50000)),
    grid_origin='lower_left',
    fields=['reflectivity'],
    weighting_function='Bilinear',
    roi_func='dist_from_radar_center',
    min_dist=0.0,
    max_dist=50000.0
)

print(f"Grid shape: {grid.fields['reflectivity']['data'].shape}")
```

### Sector Conversion

Convert sector to Cartesian grid:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Convert sector
grid = pyart.map.grid_from_sector(
    radar, 
    center_angle=90.0, 
    width_angle=45.0,
    range_1_km=0.0,
    range_2_km=50.0,
    grid_shape=(101, 101),
    fields=['reflectivity']
)

print(f"Sector grid shape: {grid.fields['reflectivity']['data'].shape}")
```

## Advanced Conversion

### Custom Grid Limits

Specify custom grid limits:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Convert with custom limits
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(201, 201),
    grid_limits=(( -25000, 25000), ( -25000, 25000)),
    fields=['reflectivity']
)

print(f"Grid shape: {grid.fields['reflectivity']['data'].shape}")
```

### Multiple Fields

Convert multiple fields:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Convert multiple fields
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity', 'velocity', 'spectrum_width']
)

print(f"Available fields: {list(grid.fields.keys())}")
```

### Weighting Functions

Use different weighting functions:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Nearest neighbor
grid_nn = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity'],
    weighting_function='Nearest'
)

# Bilinear
grid_bl = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity'],
    weighting_function='Bilinear'
)

# Cressman
grid_cr = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity'],
    weighting_function='Cressman'
)
```

## ROI Functions

### Distance from Radar Center

Limit conversion to specific distance range:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Convert with distance ROI
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity'],
    roi_func='dist_from_radar_center',
    min_dist=10000.0,
    max_dist=40000.0
)

print(f"Grid shape: {grid.fields['reflectivity']['data'].shape}")
```

### Custom ROI Function

Define custom ROI function:

```python
import pyart

def custom_roi(x, y):
    """Custom ROI function."""
    # x, y are grid coordinates
    # Return True if point should be included
    return (x**2 + y**2) < (30000.0)**2

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Convert with custom ROI
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity'],
    roi_func=custom_roi
)

print(f"Grid shape: {grid.fields['reflectivity']['data'].shape}")
```

## Height Conversion

### CAPPI (Constant Altitude PPI)

Convert to constant altitude:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Convert to CAPPI at 2 km altitude
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity'],
    edge_factor=0.0
)

# Write to netCDF
pyart.io.write_arm_netcdf(grid, 'cappi_2km.nc')

print("CAPPI created successfully")
```

### RHI (Range-Height Indicator)

Convert to range-height:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Convert to RHI
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 501),
    fields=['reflectivity'],
    edge_factor=0.0
)

# Write to netCDF
pyart.io.write_arm_netcdf(grid, 'rhi.nc')

print("RHI created successfully")
```

## Best Practices

### 1. Choose Appropriate Grid Size

```python
# Good: Appropriate grid size
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity']
)

# Bad: Too large grid size
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(1001, 1001),
    fields=['reflectivity']
)
```

### 2. Use Appropriate Weighting

```python
# For discrete data
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity'],
    weighting_function='Nearest'
)

# For continuous data
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity'],
    weighting_function='Bilinear'
)
```

### 3. Check Grid Coverage

```python
# Convert to grid
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity']
)

# Check for missing data
reflectivity = grid.fields['reflectivity']['data']
missing_fraction = np.sum(np.isnan(reflectivity)) / reflectivity.size

print(f"Missing data fraction: {missing_fraction:.3f}")
```

### 4. Use ROI for Efficiency

```python
# Use ROI to limit conversion area
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity'],
    roi_func='dist_from_radar_center',
    min_dist=0.0,
    max_dist=30000.0
)
```

## References

- Py-ART Documentation: https://arm-doe.github.io/Py-ART/
- ARM Radar Handbook: https://www.arm.gov/publications/handbooks/radar_handbook.pdf