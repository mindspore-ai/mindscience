---
name: wrf-python
description: Diagnostic and interpolation routines for WRF-ARW model output. Use when working with Weather Research and Forecasting (WRF-ARW) model data for:(1) Computing meteorological diagnostics (CAPE, storm relative helicity, etc.), (2) Interpolating model output to different grids, (3) Creating cross sections and vertical profiles, (4) Mapping between model levels and pressure levels, (5) Plotting model data with cartopy/basemap/PyNGL, (6) Processing netCDF files, or (7) Analyzing model output for verification and visualization
license: Apache-2.0
metadata:
    skill-author: K-Dense Inc.
---

# WRF-Python

Diagnostic and interpolation routines for WRF-ARW model output.

## Overview

WRF-Python provides over 30 diagnostic calculations, several interpolation routines, and utilities to help with plotting via cartopy, basemap, or PyNGL. The functionality is similar to what is provided by NCL WRF package for working with Weather Research and Forecasting (WRF-ARW) model output.

## Quick Start

**Installation:**
```bash
conda install -c conda-forge wrf-python
# or
pip install wrf-python
```

**Basic diagnostic calculation:**
```python
import numpy as np
from wrf import getvar, getvar_base

# Load WRF netCDF file
ncfile = 'wrfout_d01.nc'

# Get temperature on pressure levels
t = getvar(ncfile, 'T', True)  # True = use unpivoted data
print(f"Temperature shape: {t.shape}")
```

**Basic interpolation:**
```python
from wrf import vertcross, geobucket

# Interpolate to pressure levels
t_p = getvar(ncfile, 'T', True)

# Get geopotential height
h = getvar(ncfile, 'GHT', True)

# Interpolate to model levels
t_ml = vertcross(t_p, h)
print(f"Temperature on model levels: {t_ml.shape}")
```

## Core Workflow

### 1. Loading WRF Output

**NetCDF file structure:**
```python
import netCDF4 as nc

# Open netCDF file
ncfile = nc.Dataset('wrfout_d01.nc', 'r')

# List available variables
print("Available variables:")
for var in ncfile.variables:
    print(f"  {var}")
```

**Common WRF variables:**
- `T`: Temperature (K)
- `U`, `V`: Wind components (m/s)
- `W`: Vertical velocity (m/s)
- `P`: Pressure (Pa)
- `QVAPOR`: Water vapor mixing ratio (kg/kg)
- `QRAIN`: Rain mixing ratio (kg/kg)
- `QSNOW`: Snow mixing ratio (kg/kg)
- `QGRAUP`: Graupel mixing ratio (kg/kg)
- `QICE`: Ice mixing ratio (kg/kg)
- `REFLTC`: Reflectivity longwave
- `REFLTM`: Reflectivity (1.6 µm)
- `REFLTS`: Reflectivity shortwave
- `GHT`: Geopotential height (m)
- `PH`: Perturbation geopotential height (m)
- `Z`: Terrain height (m)

### 2. Getting Variables

**Basic variable retrieval:**
```python
from wrf import getvar

# Get temperature on pressure levels
t = getvar(ncfile, 'T', True)  # True = unpivoted
print(f"Temperature shape: {t.shape}")  # (time, level, lat, lon)
```

**Get multiple variables:**
```python
# Get multiple variables
u = getvar(ncfile, 'U', True)  # U wind component
v = getvar(ncfile, 'V', True)  # V wind component
w = getvar(ncfile, 'W', True)  # W wind component
```

**Specific time level:**
```python
# Get specific time step
t = getvar(ncfile, 'T', True, timeidx=0)

# Get specific pressure level
t = getvar(ncfile, 'T', True, lev=500)
```

**Base state:**
```python
from wrf import getvar_base

# Get base state (unperturbed)
t_base = getvar_base(ncfile, 'T', True)
```

### 3. Diagnostic Calculations

**Available diagnostics:**
- CAPE (Convective Available Potential Energy)
- Storm relative helicity
- Equivalent potential temperature
- Brunt-Väisäla frequency
- Bulk Richardson number
- Moisture flux convergence
- Precipitable water
- Integrated vapor transport
- U, V, W wind components
- Wind speed and direction
- Relative humidity
- Potential temperature
- Equivalent potential vorticity
- etc.

**CAPE calculation:**
```python
from wrf import getcape

# Calculate CAPE
cape = getcape(ncfile, True)  # True = use unpivoted data
print(f"CAPE shape: {cape.shape}")
```

**Storm relative helicity:**
```python
from wrf import srhel

# Calculate storm relative helicity
srh = srhel(ncfile, True)
print(f"Storm relative helicity shape: {srh.shape}")
```

**Wind speed:**
```python
from wrf import uvmet

# Calculate wind speed
spd = uvmet(ncfile, 'uvmet', True)
print(f"Wind speed shape: {spd.shape}")
```

**Multiple diagnostics:**
```python
# Calculate several diagnostics
cape = getcape(ncfile, True)
srh = srhel(ncfile, True)
spd = uvmet(ncfile, 'uvmet', True)
```

### 4. Interpolation

**Vertical interpolation:**
```python
from wrf import vertcross

# Get variables on pressure levels
t_p = getvar(ncfile, 'T', True)
h = getvar(ncfile, 'GHT', True)

# Interpolate to model levels
t_ml = vertcross(t_p, h)
print(f"Temperature on model levels: {t_ml.shape}")
```

**Geopotential height interpolation:**
```python
from wrf import ght

# Get geopotential height
ght = ght(ncfile, True)
```

**Geobucket:**
```python
from wrf import geobucket

# Geobucket interpolation
# See references/interpolation.md for details
```

### 5. Mapping Between Levels

**Pressure to model levels:**
```python
from wrf import getvar, vertcross

# Get pressure level data
t_p = getvar(ncfile, 'T', True)
h = getvar(ncfile, 'GHT', True)

# Interpolate to model levels
t_ml = vertcross(t_p, h)
```

**Model to pressure levels:**
```python
from wrf import pres

# Get pressure level data from model levels
# See references/interpolation.md for details
```

## Common Applications

### Diagnostic Analysis

**CAPE calculation:**
```python
from wrf import getcape

# Calculate CAPE for severe weather analysis
cape = getcape(ncfile, True)

# Find high CAPE regions
high_cape = cape > 2000  # J/kg
print(f"High CAPE grid points: {np.sum(high_cape)}")
```

**Storm identification:**
```python
from wrf import srhel

# Storm relative helicity
srh = srhel(ncfile, True)

# Identify storm regions
storm = srh > 1.0
print(f"Storm grid points: {np.sum(storm)}")
```

**Wind analysis:**
```python
from wrf import uvmet, uvmet10

# Wind speed
spd = uvmet(ncfile, 'uvmet', True)

# 10-meter wind speed
spd10 = uvmet10(ncfile, 'uvmet10', True)
```

### Cross Sections

**Vertical cross section:**
```python
from wrf import vertcross

# Get variables
t_p = getvar(ncfile, 'T', True)
h = getvar(ncfile, 'GHT', True)

# Interpolate to model levels
t_ml = vertcross(t_p, h)

# Extract cross section at specific location
lat_idx = 50  # Latitude index
lon_idx = 50  # Longitude index

cross_section = t_ml[:, :, lat_idx, lon_idx]
print(f"Cross section shape: {cross_section.shape}")
```

**Horizontal cross section:**
```python
# Extract horizontal cross section
time_idx = 0
level_idx = 0

horizontal_section = t_p[time_idx, level_idx, :, :]
print(f"Horizontal section shape: {horizontal_section.shape}")
```

### Time Series

**Time series at point:**
```python
# Get time series at specific location
lat_idx = 50
lon_idx = 50

t_series = t[:, :, lat_idx, lon_idx]
print(f"Time series shape: {t_series.shape}")
```

**Time series averaged over region:**
```python
# Average over spatial region
lat_slice = slice(40, 60)
lon_slice = slice(40, 60)

t_avg = t[:, :, lat_slice, lon_slice].mean(axis=(2, 3))
print(f"Averaged time series shape: {t_avg.shape}")
```

## Best Practices

### 1. Memory Management
- Use `xarray` for large files if available
- Close netCDF files after reading
- Process variables in chunks for very large datasets
- Use `getvar_base` for base state comparisons

### 2. Coordinate Handling
- WRF uses lat-lon coordinates
- Be aware of grid staggering
- Check domain boundaries
- Use appropriate interpolation for grid mismatches

### 3. Units
- Temperature: K
- Pressure: Pa
- Wind: m/s
- Geopotential height: m
- Mixing ratios: kg/kg
- CAPE: J/kg

### 4. Performance
- Use unpivoted data when possible (faster)
- Process multiple variables together
- Use vectorized operations
- Consider parallel processing for large datasets

### 5. Data Validation
- Check for missing values
- Verify coordinate ranges
- Validate physical ranges (e.g., T > 200 K)
- Check array shapes and dimensions

## Resources

### Scripts

**`scripts/template_diagnostic.py`**
Basic diagnostic calculation template.

**`scripts/template_interpolation.py`**
Variable interpolation template.

**`scripts/template_cross_section.py`**
Cross section extraction template.

**`scripts/template_timeseries.py`**
Time series analysis template.

### References

- **`references/diagnostics.md`** - Complete list of available diagnostics
- **`references/interpolation.md`** - Interpolation routines and mapping
- **`references/plotting.md`** - Plotting with cartopy, basemap, PyNGL
- **`references/coordinate_systems.md`** - WRF coordinate systems and grids
- **`references/advanced_usage.md`** - Advanced features and performance tips

## Common Issues

**NetCDF file not openable:**
- Check file path and permissions
- Verify file format is WRF netCDF
- Check for file corruption

**Variable not found:**
- Verify variable name in netCDF file
- Check variable spelling and case sensitivity
- Ensure variable is in the correct domain

**Memory errors:**
- Use `xarray` for large files
- Process in smaller chunks
- Close files after reading
- Reduce spatial resolution

**Incorrect array shapes:**
- Check coordinate dimensions
- Verify pressure level structure
- Check time dimension
- Review WRF model configuration

**Interpolation errors:**
- Ensure geopotential height is available
- Check coordinate alignment
- Verify interpolation method
- Review grid staggering

## Additional Resources

- Official documentation: https://wrf-python.rtfd.org/
- GitHub repository: https://github.com/NCAR/wrf-python
- Citation: https://wrf-python.readthedocs.io/en/latest/citation.html
- FAQ: https://wrf-python.readthedocs.io/en/latest/faq.html
- Support: https://github.com/NCAR/wrf-python/issues
