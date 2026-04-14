#!/usr/bin/env python3
"""
Basic diagnostic calculation template for WRF-Python.

Usage:
    python scripts/template_diagnostic.py
"""

import numpy as np
from wrf import getvar, getcape, srhel, uvmet

# Load WRF netCDF file
ncfile = 'wrfout_d01.nc'

# Get temperature on pressure levels
t = getvar(ncfile, 'T', True)  # True = use unpivoted data
print(f"Temperature shape: {t.shape}")  # (time, level, lat, lon)

# Get wind components
u = getvar(ncfile, 'U', True)
v = getvar(ncfile, 'V', True)
w = getvar(ncfile, 'W', True)

# Calculate CAPE
cape = getcape(ncfile, True)
print(f"CAPE shape: {cape.shape}")

# Calculate storm relative helicity
srh = srhel(ncfile, True)
print(f"Storm relative helicity shape: {srh.shape}")

# Calculate wind speed
spd = uvmet(ncfile, 'uvmet', True)
print(f"Wind speed shape: {spd.shape}")

# Print diagnostic ranges
print(f"\nTemperature range: [{t.min():.1f}, {t.max():.1f}] K")
print(f"CAPE range: [{cape.min():.1f}, {cape.max():.1f}] J/kg")
print(f"SRH range: [{srh.min():.3f}, {srh.max():.3f}]")
print(f"Wind speed range: [{spd.min():.2f}, {spd.max():.2f}] m/s")

# Identify high CAPE regions
high_cape = cape > 2000  # J/kg
print(f"\nHigh CAPE grid points: {np.sum(high_cape)}")
print(f"Max CAPE: {cape.max():.1f} J/kg")

# Identify storm regions
storm = srh > 1.0
print(f"Storm grid points: {np.sum(storm)}")
print(f"Max SRH: {srh.max():.3f}")

# Identify strong winds
strong_wind = spd > 20.0  # m/s
print(f"Strong wind grid points: {np.sum(strong_wind)}")
print(f"Max wind speed: {spd.max():.2f} m/s")
