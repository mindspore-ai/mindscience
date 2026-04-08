#!/usr/bin/env python3
"""
Calculate statistical summaries of GRIB data.

Usage:
    python calculate_statistics.py <input.grib> [--variable <var>] [--spatial] [--temporal]
    
Examples:
    python calculate_statistics.py data.grib
    python calculate_statistics.py data.grib --variable t2m
    python calculate_statistics.py data.grib --spatial --temporal
"""

import sys
import argparse
import xarray as xr
import numpy as np


def calculate_global_statistics(ds, variable=None):
    """Calculate global statistics for dataset."""
    
    if variable:
        if variable not in ds.data_vars:
            print(f"Error: Variable '{variable}' not found")
            return None
        vars_to_calc = [variable]
    else:
        vars_to_calc = list(ds.data_vars.keys())
    
    print("\n=== Global Statistics ===")
    
    for var_name in vars_to_calc:
        var = ds[var_name]
        
        print(f"\n{var_name}:")
        print(f"  Mean: {float(var.mean()):.6f}")
        print(f"  Std: {float(var.std()):.6f}")
        print(f"  Min: {float(var.min()):.6f}")
        print(f"  Max: {float(var.max()):.6f}")
        print(f"  Median: {float(var.median()):.6f}")
        print(f"  25th percentile: {float(var.quantile(0.25)):.6f}")
        print(f"  75th percentile: {float(var.quantile(0.75)):.6f}")
        print(f"  Size: {var.size}")
        print(f"  Missing values: {int(np.isnan(var).sum())}")


def calculate_spatial_statistics(ds, variable=None):
    """Calculate spatial statistics."""
    
    if variable:
        if variable not in ds.data_vars:
            print(f"Error: Variable '{variable}' not found")
            return None
        vars_to_calc = [variable]
    else:
        vars_to_calc = list(ds.data_vars.keys())
    
    print("\n=== Spatial Statistics ===")
    
    for var_name in vars_to_calc:
        var = ds[var_name]
        
        # Check if spatial dimensions exist
        spatial_dims = [d for d in var.dims if d in ['latitude', 'longitude']]
        
        if spatial_dims:
            print(f"\n{var_name}:")
            
            # Spatial mean
            spatial_mean = var.mean(dim=spatial_dims)
            print(f"  Spatial mean: {float(spatial_mean.mean()):.6f}")
            
            # Spatial std
            spatial_std = var.std(dim=spatial_dims)
            print(f"  Spatial std: {float(spatial_std.mean()):.6f}")
            
            # Zonal mean (average over longitudes)
            if 'longitude' in var.dims:
                zonal_mean = var.mean(dim='longitude')
                print(f"  Zonal mean range: {float(zonal_mean.min()):.6f} to {float(zonal_mean.max()):.6f}")
            
            # Meridional mean (average over latitudes)
            if 'latitude' in var.dims:
                meridional_mean = var.mean(dim='latitude')
                print(f"  Meridional mean range: {float(meridional_mean.min()):.6f} to {float(meridional_mean.max()):.6f}")


def calculate_temporal_statistics(ds, variable=None):
    """Calculate temporal statistics."""
    
    if variable:
        if variable not in ds.data_vars:
            print(f"Error: Variable '{variable}' not found")
            return None
        vars_to_calc = [variable]
    else:
        vars_to_calc = list(ds.data_vars.keys())
    
    print("\n=== Temporal Statistics ===")
    
    for var_name in vars_to_calc:
        var = ds[var_name]
        
        # Check if time dimension exists
        if 'time' in var.dims:
            print(f"\n{var_name}:")
            
            # Time mean
            time_mean = var.mean(dim='time')
            print(f"  Time mean: {float(time_mean.mean()):.6f}")
            
            # Time std
            time_std = var.std(dim='time')
            print(f"  Time std: {float(time_std.mean()):.6f}")
            
            # Trend (linear)
            if len(var.coords['time']) > 1:
                time_numeric = (var.coords['time'] - var.coords['time'][0]).dt.days
                flattened = var.values.flatten()
                if len(time_numeric) == len(flattened):
                    # This is simplified - proper implementation would handle dimensions
                    print(f"  Time dimension: {len(var.coords['time'])} timesteps")


def main():
    parser = argparse.ArgumentParser(description='Calculate statistics of GRIB data')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('--variable', help='Variable name')
    parser.add_argument('--spatial', action='store_true', help='Calculate spatial statistics')
    parser.add_argument('--temporal', action='store_true', help='Calculate temporal statistics')
    
    args = parser.parse_args()
    
    try:
        with xr.open_dataset(args.input_file, engine='cfgrib') as ds:
            calculate_global_statistics(ds, args.variable)
            
            if args.spatial:
                calculate_spatial_statistics(ds, args.variable)
            
            if args.temporal:
                calculate_temporal_statistics(ds, args.variable)
    
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()