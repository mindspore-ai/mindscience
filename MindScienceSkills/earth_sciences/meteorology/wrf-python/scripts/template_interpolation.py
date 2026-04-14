#!/usr/bin/env python3
"""
Variable interpolation template for WRF-Python.

Usage:
    python scripts/template_interpolation.py
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

print(f"Temperature on pressure levels: {t_p.shape}")
print(f"Temperature on model levels: {t_ml.shape}")

# Get pressure level data
p = getvar(ncfile, 'P', True)
print(f"Pressure levels: {p.shape}")

# Extract cross section at specific location
lat_idx = 50  # Latitude index
lon_idx = 50  # Longitude index

cross_section = t_ml[:, :, lat_idx, lon_idx]
print(f"Cross section shape: {cross_section.shape}")

# Extract time series at point
time_series = t_ml[:, :, lat_idx, lon_idx]
print(f"Time series shape: {time_series.shape}")

print(f"\nCross section values: {cross_section}")
print(f"Time series values: {time_series}")
