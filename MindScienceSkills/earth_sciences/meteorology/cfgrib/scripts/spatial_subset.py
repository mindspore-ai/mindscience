#!/usr/bin/env python3
"""
Extract spatial subsets from GRIB files.

Usage:
    python spatial_subset.py <input.grib> <output.nc> --lat-north <lat> --lat-south <lat> --lon-west <lon> --lon-east <lon>
    
Examples:
    python spatial_subset.py input.grib output.nc --lat-north 60 --lat-south 30 --lon-west -120 --lon-east -90
"""

"""

import sys
import argparse
import xarray as xr


def extract_spatial_subset(input_file, output_file, lat_north, lat_south, lon_west, lon_east):
    """Extract spatial subset from GRIB file."""
    
    try:
        with xr.open_dataset(input_file, engine='cfgrib') as ds:
            # Select spatial region
            subset = ds.sel(
                latitude=slice(lat_north, lat_south),
                longitude=slice(lon_west, lon_east)
            )
            
            # Write to NetCDF
            subset.to_netcdf(output_file)
            
            print(f"Subset extracted successfully")
            print(f"Input shape: {dict(ds.dims)}")
            print(f"Output shape: {dict(subset.dims)}")
            print(f"Output file: {output_file}")
            
            return True
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Extract spatial subset from GRIB files')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('output_file', help='Output NetCDF file')
    parser.add_argument('--lat-north', type=float, required=True, help='Northern latitude boundary')
    parser.add_argument('--lat-south', type=float, required=True, help='Southern latitude boundary')
    parser.add_argument('--lon-west', type=float, required=True, help='Western longitude boundary')
    parser.add_argument('--lon-east', type=float, required=True, help='Eastern longitude boundary')
    
    args = parser.parse_args()
    
    success = extract_spatial_subset(args.input_file, args.output_file, 
                                    args.lat_north, args.lat_south, 
                                    args.lon_west, args.lon_east)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()