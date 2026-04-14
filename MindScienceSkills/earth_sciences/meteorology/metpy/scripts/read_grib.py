#!/usr/bin/env python3
"""
Basic MetPy example: Read and plot GRIB2 data
"""

from metpy.io import read_gribber
from metpy.plots import declarative

def main():
    # Read GRIB2 file
    data = read_gribber('data.grib2')
    
    # Create declarative plot
    declarative.plot(data, 'temperature')

if __name__ == '__main__':
    main()
