#!/usr/bin/env python3
"""
RGB composite creation template for Satpy.

Usage:
    python scripts/template_composite.py
"""

from satpy import Scene

# Load satellite file
scn = Scene('satellite_file.nc')

# Load RGB bands
scn.load(['red_band', 'green_band', 'blue_band'])

# Create RGB composite
rgb = scn['RGB']

# Save to PNG
rgb.save('rgb_image.png')

print("RGB composite saved to rgb_image.png")
print(f"RGB shape: {rgb.shape}")
