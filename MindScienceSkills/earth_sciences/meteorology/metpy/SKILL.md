---
name: metpy
description: Python tools for reading, visualizing, and performing calculations with weather data. Use when working with meteorological data, weather maps, atmospheric calculations, or GEMPAK-like functionality in Python. Supports multiple data formats, grid operations, and integration with scientific Python ecosystem.
---

# MetPy

MetPy is a collection of Python tools for reading, visualizing, and performing calculations with weather data.

## Quick Start

### Reading Data

```python
from metpy.io import read_grib

# Read GRIB2 file
data = read_grib('data.grib2')

# Access data array
temperature = data['temperature']
```

### Basic Plotting

```python
from metpy.plots import declarative
from metpy.io import read_grib

# Read data
data = read_grib('data.grib2')

# Create declarative plot
declarative.plot(data, 'temperature', clevs=np.arange(250, 310, 5))
```

### Grid Calculations

```python
from metpy.calc import advection
from metpy.io import read_grib

# Read data
data = read_grib('data.grib2')

# Calculate advection
advected = advection.advection(data['u'], data['v'], data['temperature'])
```

## Core Concepts

### Data Structures

MetPy uses xarray DataArrays for gridded data:

```python
import xarray as xr
from metpy.io import read_grib

# Read data
data = read_grib('data.grib2')

# Access as xarray
temperature = data['temperature']
print(temperature.dims)  # Dimensions
print(temperature.coords)  # Coordinates
```

### Grid Types

**Latitude-Longitude Grids**: Regular lat-lon grids
```python
from metpy.io import read_grib

data = read_grib('data.grib2')
# Automatically recognized as lat-lon grid
```

**Projected Grids**: Various map projections
```python
from metpy.io import read_grib

data = read_grib('data.grib2')
# Projection information from GRIB metadata
```

### Coordinate Systems

MetPy supports multiple coordinate systems:
- Latitude/Longitude
- Map projections (via pyproj)
- Grid indices
- Cross sections

## Data I/O

### Reading GRIB Files

```python
from metpy.io import read_grib

# Read GRIB2 file
data = read_grib('data.grib2')

# Read specific variable
temperature = read_grib('data.grib2', var_name='temperature')
```

### Reading NetCDF Files

```python
from metpy.io import read_netcdf

# Read NetCDF file
data = read_netcdf('data.nc')

# Read specific variable
temperature = read_netcdf('data.nc', var_name='temperature')
```

### Writing Data

```python
from metpy.io import write_grib

# Write to GRIB2
write_grib(data, 'output.grib2')

# Write specific variable
write_grib(data['temperature'], 'output.grib2')
```

## Calculations

### Thermodynamic Calculations

```python
from metpy.calc import thermo

# Calculate potential temperature
theta = thermo.potential_temperature(temperature, pressure)

# Calculate equivalent potential temperature
theta_e = thermo.equivalent_potential_temperature(temperature, pressure)
```

### Kinematic Calculations

```python
from metpy.calc import kinematics

# Calculate wind speed
speed = kinematics.wind_speed(u, v)

# Calculate wind direction
direction = kinematics.wind_direction(u, v)
```

### Dynamic Calculations

```python
from metpy.calc import dynamics

# Calculate vorticity
vorticity = dynamics.vorticity(u, v)

# Calculate divergence
divergence = dynamics.divergence(u, v)
```

### Advection

```python
from metpy.calc import advection

# Advect scalar field
advected = advection.advection(u, v, scalar_field)
```

## Plotting

### Declarative Plotting

```python
from metpy.plots import declarative

# Create simple plot
declarative.plot(data, 'temperature')

# Custom contour levels
declarative.plot(data, 'temperature', clevs=np.arange(250, 310, 5))
```

### Cross Sections

```python
from metpy.plots import cross_section

# Create cross section plot
cross_section.plot(data, 'temperature', lat=40, lon=-100)
```

### Hodographs

```python
from metpy.plots import hodograph

# Create hodograph
hodograph.plot(u, v)
```

### Skew-T Plots

```python
from metpy.plots import skewt

# Create skew-T plot
skewt.plot(temperature, pressure)
```

## Grid Operations

### Grid Manipulation

```python
from metpy.calc import grid

# Calculate grid spacing
dx, dy = grid.grid_spacing(data)

# Calculate grid area
area = grid.grid_area(data)
```

### Interpolation

```python
from metpy.calc import interpolation

# Interpolate to new grid
interpolated = interpolation.interpolate(data, new_grid)
```

### Masking

```python
from metpy.calc import masking

# Create land mask
mask = masking.land_mask(data)

# Apply mask
masked_data = data.where(mask)
```

## Resources

- **Data I/O**: See [data_io.md](references/data_io.md)
- **Calculations**: See [calculations.md](references/calculations.md)
- **Plotting**: See [plotting.md](references/plotting.md)
- **Grid operations**: See [grid_operations.md](references/grid_operations.md)
- **Advanced features**: See [advanced.md](references/advanced.md)
