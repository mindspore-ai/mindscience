#!/usr/bin/env python3
"""
Extract temporal subsets from GRIB files.

Usage:
    python temporal_subset.py <input.grib> <output.nc> --start <date> --end <date>
    
Examples:
    python temporal_subset.py input.grib output.nc --start "2024-01-01" --end "2024-01-31"
"""

import sys
import argparse
import xarray as xr


def extract_temporal_subset(input_file, output_file, start_date, end_date):
    """Extract temporal subset from GRIB file."""
    
    try:
        with xr.open_dataset(input_file, engine='cfgrib') as ds:
            # Check if time dimension exists
            if 'time' not in ds.coords:
                print("Error: Dataset does not have a time dimension")
                return False
            
            # Select time range
            subset = ds.sel(time=slice(start_date, end_date))
            
            # Write to NetCDF
            subset.to_netcdf(output_file)
            
            print(f"Subset extracted successfully")
            print(f"Input time range: {ds.coords['time'].values[0]} to {ds.coords['time'].values[-1]}")
            print(f"Output time range: {subset.coords['time'].values[0]} to {subset.coords['time'].values[-1]}")
            print(f"Output file: {output_file}")
            
            return True
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Extract temporal subset from GRIB files')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('output_file', help='Output NetCDF file')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    success = extract_temporal_subset(args.input_file, args.output_file, args.start, args.end)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()