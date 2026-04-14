#!/usr/bin/env python3
"""
Basic satellite data reading template for Satpy.

Usage:
    python scripts/template_reader.py
"""

from satpy import Scene

# Load satellite file
scn = Scene('satellite_file.nc')

# Convert to xarray
ds = scn.to_xarray()

# List available datasets
print("Available datasets:")
for name in ds:
    print(f"  {name}")
    dataset = ds[name]
    print(f"    Shape: {dataset.shape}")
    print(f"    Attributes: {list(dataset.attrs.keys())}")
