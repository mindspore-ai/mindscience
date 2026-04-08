#!/usr/bin/env python3
"""
Cross section extraction template for WRF-Python.

Usage:
    python scripts/template_cross_section.py
"""

import numpy as np
from wrf import getvar, vertcross

# Load WRF netCDF file
ncfile = 'wrfout_d01.nc'

# Get temperature on pressure levels
t_p = getvar(ncfile, 'T', True)  # True = unpivoted

# Get geopotential height
h = getvar(ncfile, 'GHT', True)

# Interpolate to model levels
t_ml = vertcross(t_p, h)

# Extract cross section at specific location
lat_idx = 50  # Latitude index
lon_idx = 50  # Longitude index

cross_section = t_ml[:, :, lat_idx, lon_idx]
h_cross = h[:, :, lat_idx, lon_idx]

print(f"Cross section shape: {cross_section.shape}")
print(f"Temperature range: [{cross_section.min():.1f}, {cross_section.max():.1f}] K")
print(f"Height range: [{h_cross.min():.1f}, {h_cross.max():.1f}] m")

# Extract horizontal cross section
time_idx = 0  # Time index
level_idx = 10  # Pressure level index

horizontal_section = t_p[time_idx, level_idx, :, :]
print(f"Horizontal section shape: {horizontal_section.shape}")
