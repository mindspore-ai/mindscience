#!/usr/bin/env python3
"""
Time series analysis template for WRF-Python.

Usage:
    python scripts/template_timeseries.py
"""

import numpy as np
from wrf import getvar

# Load WRF netCDF file
ncfile = 'wrfout_d01.nc'

# Get temperature on pressure levels
t = getvar(ncfile, 'T', True)  # True = unpivoted
print(f"Temperature shape: {t.shape}")  # (time, level, lat, lon)

# Get coordinates
lat = getvar(ncfile, 'XLAT', False)
lon = getvar(ncfile, 'XLON', False)

# Extract time series at specific location
lat_idx = 50  # Latitude index
lon_idx = 50  # Longitude index
level_idx = 10  # Pressure level index

# Time series at point
t_point = t[:, level_idx, lat_idx, lon_idx]
print(f"Time series shape at point: {t_point.shape}")

# Extract time series averaged over region
lat_slice = slice(40, 60)  # Latitude range
lon_slice = slice(40, 60)  # Longitude range

# Average over spatial region
t_avg = t[:, level_idx, lat_slice, lon_slice].mean(axis=(2, 3))
print(f"Averaged time series shape: {t_avg.shape}")

# Calculate statistics
print(f"\nTime series statistics:")
print(f"  Mean: {t_point.mean():.2f} K")
print(f"  Std: {t_point.std():.2f} K")
print(f"  Min: {t_point.min():.2f} K")
print(f"  Max: {t_point.max():.2f} K")

# Get wind speed
from wrf import uvmet
spd = uvmet(ncfile, 'uvmet', True)
spd_point = spd[:, level_idx, lat_idx, lon_idx]

print(f"\nWind speed at point:")
print(f"  Mean: {spd_point.mean():.2f} m/s")
print(f"  Max: {spd_point.max():.2f} m/s")

# Get CAPE
from wrf import getcape
cape = getcape(ncfile, True)
cape_point = cape[:, level_idx, lat_idx, lon_idx]

print(f"\nCAPE at point:")
print(f"  Mean: {cape_point.mean():.1f} J/kg")
print(f"  Max: {cape_point.max():.1f} J/kg")
