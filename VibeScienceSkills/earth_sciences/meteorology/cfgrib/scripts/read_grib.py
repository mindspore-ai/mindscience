#!/usr/bin/env python3
"""
Read and display GRIB file contents as xarray Dataset.

Usage:
    python read_grib.py <input.grib> [--summary] [--variables] [--coords] [--info]
    
Examples:
    python read_grib.py data.grib
    python read_grib.py data.grib --summary
    python read_grib.py data.grib --variables --coords
"""

import sys
import argparse
import xarray as xr


def print_summary(ds):
    """Print summary of the dataset."""
    print("\n=== Dataset Summary ===")
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Number of variables: {len(ds.data_vars)}")
    print(f"Number of coordinates: {len(ds.coords)}")
    print(f"Number of attributes: {len(ds.attrs)}")
    
    if 'time' in ds.coords:
        print(f"\nTime range:")
        print(f"  Start: {ds.coords['time'].values[0]}")
        print(f"  End: {ds.coords['time'].values[-1]}")
        print(f"  Number of timesteps: {len(ds.coords['time'])}")


def print_variables(ds):
    """Print information about data variables."""
    print("\n=== Data Variables ===")
    
    for var_name, var in ds.data_vars.items():
        print(f"\n{var_name}:")
        print(f"  Dimensions: {var.dims}")
        print(f"  Shape: {var.shape}")
        print(f"  Size: {var.size}")
        print(f"  Dtype: {var.dtype}")
        
        if 'long_name' in var.attrs:
            print(f"  Long name: {var.attrs['long_name']}")
        if 'units' in var.attrs:
            print(f"  Units: {var.attrs['units']}")


def print_coords(ds):
    """Print information about coordinates."""
    print("\n=== Coordinates ===")
    
    for coord_name, coord in ds.coords.items():
        print(f"\n{coord_name}:")
        print(f"  Dimensions: {coord.dims}")
        print(f"  Shape: {coord.shape}")
        print(f"  Size: {coord.size}")
        print(f"  Dtype: {coord.dtype}")
        
        if coord.size <= 10:
            print(f"  Values: {coord.values}")
        else:
            print(f"  First 5 values: {coord.values[:5]}")
            print(f"  Last 5 values: {coord.values[-5:]}")


def print_info(ds):
    """Print detailed information about the dataset."""
    print("\n=== Detailed Information ===")
    print(ds.info())


def main():
    parser = argparse.ArgumentParser(description='Read GRIB file as xarray Dataset')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('--summary', action='store_true', help='Print dataset summary')
    parser.add_argument('--variables', action='store_true', help='Print variable information')
    parser.add_argument('--coords', action='store_true', help='Print coordinate information')
    parser.add_argument('--info', action='store_true', help='Print detailed info')
    
    args = parser.parse_args()
    
    try:
        with xr.open_dataset(args.input_file, engine='cfgrib') as ds:
            if not (args.summary or args.variables or args.coords or args.info):
                args.summary = True
                args.variables = True
                args.coords = True
            
            if args.summary:
                print_summary(ds)
            
            if args.variables:
                print_variables(ds)
            
            if args.coords:
                print_coords(ds)
            
            if args.info:
                print_info(ds)
    
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()