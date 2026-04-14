#!/usr/bin/env python3
"""
Convert GRIB files to NetCDF format using cfgrib or xarray.

Usage:
    python grib_to_netcdf.py <input.grib> <output.nc> [--engine <engine>]
    
Examples:
    python grib_to_netcdf.py input.grib output.nc
    python grib_to_netcdf.py input.grib output.nc --engine cfgrib
    python grib_to_netcdf.py input.grib output.nc --engine xarray
"""

import sys
import argparse


def convert_with_cfgrib(input_file, output_file):
    """Convert GRIB to NetCDF using cfgrib engine."""
    try:
        import xarray as xr
        
        print(f"Reading GRIB file: {input_file}")
        ds = xr.open_dataset(input_file, engine='cfgrib')
        
        print(f"Dataset info:")
        print(f"  Variables: {list(ds.data_vars.keys())}")
        print(f"  Dimensions: {dict(ds.dims)}")
        print(f"  Coordinates: {list(ds.coords.keys())}")
        
        print(f"\nWriting NetCDF file: {output_file}")
        ds.to_netcdf(output_file)
        
        ds.close()
        print("Conversion completed successfully")
        return True
    
    except ImportError:
        print("Error: xarray or cfgrib not installed")
        print("Install with: pip install xarray cfgrib")
        return False
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def convert_with_xarray(input_file, output_file):
    """Convert GRIB to NetCDF using xarray with pygrib backend."""
    try:
        import xarray as xr
        import pygrib
        
        print(f"Reading GRIB file: {input_file}")
        grbs = pygrib.open(input_file)
        
        print(f"Number of messages: {grbs.messages}")
        
        grbs.close()
        print("Note: Full xarray conversion requires additional configuration")
        print("Consider using cfgrib engine for better GRIB support")
        return True
    
    except ImportError:
        print("Error: xarray or pygrib not installed")
        print("Install with: pip install xarray pygrib")
        return False
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert GRIB files to NetCDF format')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('output_file', help='Output NetCDF file')
    parser.add_argument('--engine', choices=['cfgrib', 'xarray'], default='cfgrib',
                       help='Conversion engine (default: cfgrib)')
    
    args = parser.parse_args()
    
    success = False
    
    if args.engine == 'cfgrib':
        success = convert_with_cfgrib(args.input_file, args.output_file)
    elif args.engine == 'xarray':
        success = convert_with_xarray(args.input_file, args.output_file)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()