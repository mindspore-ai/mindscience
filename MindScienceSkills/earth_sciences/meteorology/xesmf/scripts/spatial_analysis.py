#!/usr/bin/env python3
"""
Perform spatial operations on large datasets.

Usage:
    python spatial_analysis.py <input_file> --operation <operation> --field <field>
    
Examples:
    python spatial_analysis.py large_file.nc --operation zonal_mean --field temperature
    python spatial_analysis.py large_file.nc --operation regional_mean --field temperature
"""

import sys
import argparse
import xarray as xr
import numpy as np


def zonal_mean(input_file, field_name):
    """Calculate zonal average (average over longitudes)."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks='auto')
        
        # Check if field exists
        if field_name not in ds.data_vars:
            print(f"Error: Field '{field_name}' not found")
            print(f"Available fields: {list(ds.data_vars.keys())}")
            return False
        
        # Check for spatial dimensions
        field = ds[field_name]
        if 'longitude' not in field.dims:
            print("Error: Field does not have longitude dimension")
            return False
        
        # Calculate zonal mean
        zonal_mean = field.mean(dim='longitude')
        
        # Display results
        print(f"Zonal mean calculated for {field_name}")
        print(f"Shape: {zonal_mean.shape}")
        print(f"Mean: {float(zonal_mean.mean()):.6f}")
        print(f"Min: {float(zonal_mean.min()):.6f}")
        print(f"Max: {float(zonal_mean.max()):.6f}")
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def meridional_mean(input_file, field_name):
    """Calculate meridional average (average over latitudes)."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks='auto')
        
        # Check if field exists
        if field_name not in ds.data_vars:
            print(f"Error: Field '{field_name}' not found")
            return False
        
        # Check for spatial dimensions
        field = ds[field_name]
        if 'latitude' not in field.dims:
            print("Error: Field does not have latitude dimension")
            return False
        
        # Calculate meridional mean
        meridional_mean = field.mean(dim='latitude')
        
        # Display results
        print(f"Meridional mean calculated for {field_name}")
        print(f"Shape: {meridional_mean.shape}")
        print(f"Mean: {float(meridional_mean.mean()):.6f}")
        print(f"Min: {float(meridional_mean.min()):.6f}")
        print(f"Max: {float(meridional_mean.max()):.6f}")
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def regional_mean(input_file, field_name, lat_north, lat_south, lon_west, lon_east):
    """Calculate regional average."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks='auto')
        
        # Check if field exists
        if field_name not in ds.data_vars:
            print(f"Error: Field '{field_name}' not found")
            return False
        
        # Check for spatial dimensions
        field = ds[field_name]
        if 'latitude' not in field.dims or 'longitude' not in field.dims:
            print("Error: Field does not have spatial dimensions")
            return False
        
        # Select region
        region = field.sel(
            latitude=slice(lat_north, lat_south),
            longitude=slice(lon_west, lon_east)
        )
        
        # Calculate regional mean
        regional_mean = region.mean(dim=['latitude', 'longitude'])
        
        # Display results
        print(f"Regional mean calculated for {field_name}")
        print(f"Region: {lat_north}°N to {lat_south}°N, "
              f"{lon_west}°E to {lon_east}°E")
        print(f"Mean: {float(regional_mean):.6f}")
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def spatial_std(input_file, field_name):
    """Calculate spatial standard deviation."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks='auto')
        
        # Check if field exists
        if field_name not in ds.data_vars:
            print(f"Error: Field '{field_name}' not found")
            return False
        
        # Check for spatial dimensions
        field = ds[field_name]
        if 'latitude' not in field.dims or 'longitude' not in field.dims:
            print("Error: Field does not have spatial dimensions")
            return False
        
        # Calculate spatial std
        spatial_std = field.std(dim=['latitude', 'longitude'])
        
        # Display results
        print(f"Spatial std calculated for {field_name}")
        print(f"Shape: {spatial_std.shape}")
        print(f"Mean std: {float(spatial_std.mean()):.6f}")
        print(f"Max std: {float(spatial_std.max()):.6f}")
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Spatial operations on large datasets')
    parser.add_argument('input_file', help='Input dataset file')
    parser.add_argument('--operation', choices=['zonal_mean', 'meridional_mean', 'regional_mean', 'spatial_std'], 
                       required=True, help='Spatial operation')
    parser.add_argument('--field', help='Field name to analyze')
    parser.add_argument('--lat-north', type=float, help='Northern latitude boundary')
    parser.add_argument('--lat-south', type=float, help='Southern latitude boundary')
    parser.add_argument('--lon-west', type=float, help='Western longitude boundary')
    parser.add_argument('--lon-east', type=float, help='Eastern longitude boundary')
    
    args = parser.parse_args()
    
    if args.operation == 'regional_mean':
        if args.lat_north is None or args.lat_south is None or \
           args.lon_west is None or args.lon_east is None:
            print("Error: Regional mean requires all boundary parameters")
            sys.exit(1)
        
        success = regional_mean(args.input_file, args.field, 
                                  args.lat_north, args.lat_south, 
                                  args.lon_west, args.lon_east)
    elif args.operation == 'zonal_mean':
        if args.field is None:
            print("Error: Field name required for zonal mean")
            sys.exit(1)
        success = zonal_mean(args.input_file, args.field)
    elif args.operation == 'meridional_mean':
        if args.field is None:
            print("Error: Field name required for meridional mean")
            sys.exit(1)
        success = meridional_mean(args.input_file, args.field)
    elif args.operation == 'spatial_std':
        if args.field is None:
            print("Error: Field name required for spatial std")
            sys.exit(1)
        success = spatial_std(args.input_file, args.field)
    else:
        print(f"Error: Unknown operation: {args.operation}")
        sys.exit(1)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()