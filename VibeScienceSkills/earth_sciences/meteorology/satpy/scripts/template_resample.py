#!/usr/bin/env python3
"""
Data resampling template for Satpy.

Usage:
    python scripts/template_resample.py
"""

from satpy import Scene

# Load satellite file
scn = Scene('satellite_file.nc')

# Resample to specific grid resolution
swath = scn.resample('my_area', 5000)  # 5000m resolution

# Load datasets
scn.load(['temperature', 'humidity'])

# Get resampled data
print(f"Resampled data shape: {swath['temperature'].shape}")
print(f"Resampled data shape: {swath['humidity'].shape}")
