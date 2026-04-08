#!/usr/bin/env python3
"""
Convert GRIB data to other formats (NetCDF, Zarr, CSV).

Usage:
    python convert_format.py <input.grib> <output_file> --format <format>
    
Examples:
    python convert_format.py input.grib output.nc --format netcdf
    python convert_format.py input.grib output.zarr --format zarr
    python convert_format.py input.grib output.csv --format csv
"""

import sys
import argparse
import xarray as xr


def convert_to_netcdf(input_file, output_file, compression=False):
    """Convert GRIB to NetCDF format."""
    
    try:
        with xr.open_dataset(input_file, engine='cfgrib') as ds:
            if compression:
                encoding = {var: {'zlib': True, 'complevel': 5} 
                            for var in ds.data_vars}
                ds.to_netcdf(output_file, encoding=encoding)
            else:
                ds.to_netcdf(output_file)
            
            print(f"Successfully converted to NetCDF: {output_file}")
            return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def convert_to_zarr(input_file, output_file, chunking=None):
    """Convert GRIB to Zarr format."""
    
    try:
        with xr.open_dataset(input_file, engine='cfgrib') as ds:
            if chunking:
                encoding = {var: {'chunksizes': chunking} 
                            for var in ds.data_vars}
                ds.to_zarr(output_file, encoding=encoding)
            else:
                ds.to_zarr(output_file)
            
            print(f"Successfully converted to Zarr: {output_file}")
            return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def convert_to_csv(input_file, output_file, variable=None):
    """Convert GRIB to CSV format."""
    
    try:
        with xr.open_dataset(input_file, engine='cfgrib') as ds:
            if variable:
                if variable not in ds.data_vars:
                    print(f"Error: Variable '{variable}' not found")
                    return False
                df = ds[variable].to_dataframe()
            else:
                df = ds.to_dataframe()
            
            df.to_csv(output_file)
            
            print(f"Successfully converted to CSV: {output_file}")
            return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert GRIB data to other formats')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('--format', choices=['netcdf', 'zarr', 'csv'], 
                       required=True, help='Output format')
    parser.add_argument('--compression', action='store_true', 
                       help='Enable compression (NetCDF only)')
    parser.add_argument('--variable', help='Variable to convert (CSV only)')
    
    args = parser.parse_args()
    
    success = False
    
    if args.format == 'netcdf':
        success = convert_to_netcdf(args.input_file, args.output_file, args.compression)
    elif args.format == 'zarr':
        success = convert_to_zarr(args.input_file, args.output_file)
    elif args.format == 'csv':
        success = convert_to_csv(args.input_file, args.output_file, args.variable)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()