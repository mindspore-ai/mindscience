#!/usr/bin/env python3
"""
Thermodynamic calculations example
"""

from metpy.io import read_gribber
from metpy.calc import thermo

def main():
    # Read GRIB2 file
    data = read_gribber('data.grib2')
    
    # Calculate potential temperature
    temperature = data['temperature']
    pressure = data['pressure']
    theta = thermo.potential_temperature(temperature, pressure)
    
    # Calculate equivalent potential temperature
    theta_e = thermo.equivalent_potential_temperature(temperature, pressure)
    
    print(f"Potential temperature: {theta}")
    print(f"Equivalent potential temperature: {theta_e}")

if __name__ == '__main__':
    main()
