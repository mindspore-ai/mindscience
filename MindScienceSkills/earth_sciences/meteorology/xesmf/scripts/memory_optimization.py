#!/usr/bin/env python3
"""
Optimize memory usage for large datasets.

Usage:
    python memory_optimization.py <input_file> --variable <var> --operation <op>
    
Examples:
    python memory_optimization.py large_file.nc --variable temperature --operation reduce_precision
    python memory_optimization.py large_file.nc --variable temperature --operation drop_unused
"""

import sys
import argparse
import xarray as xr
import numpy as np


def reduce_precision(input_file, variable):
    """Reduce data precision to save memory."""
    
    try:
        # Open file
        ds = xr.open_dataset(input_file, chunks='auto')
        
        # Check if variable exists
        if variable not in ds.data_vars:
            print(f"Error: Variable '{variable}' not found")
            print(f"Available variables: {list(ds.data_vars.keys())}")
            return False
        
        # Get original dtype
        original_dtype = ds[variable].dtype
        print(f"Original dtype: {original_dtype}")
        
        # Reduce precision
        if original_dtype == 'float64':
            reduced_ds = ds.astype({variable: 'float32'})
            print(f"Reduced to: float32")
        elif original_dtype == 'float32':
            reduced_ds = ds.astype({variable: 'float16'})
            print(f"Reduced to: float16")
        else:
            print(f"Dtype {original_dtype} cannot be reduced")
            return False
        
        # Calculate memory savings
        original_size = ds[variable].nbytes
        reduced_size = reduced_ds[variable].nbytes
        savings = original_size - reduced_size
        savings_percent = (savings / original_size) * 100
        
        print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
        print(f"Reduced size: {reduced_size / 1024 / 1024:.2f} MB")
        print(f"Memory savings: {savings / 1024 / 1024:.2f} MB ({savings_percent:.1f}%)")
        
        # Write reduced file
        output_file = input_file.replace('.nc', '_reduced.nc')
        reduced_ds.to_netcdf(output_file)
        
        print(f"Reduced file written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def drop_unused_variables(input_file):
    """Drop unused variables to save memory."""
    
    try:
        # Open file
        ds = xr.open_dataset(input_file, chunks='auto')
        
        # Get all variables
        all_vars = list(ds.data_vars.keys())
        print(f"Total variables: {len(all_vars)}")
        
        # Ask which variables to keep
        print("\nSelect variables to keep (comma-separated):")
        print(f"Available variables: {', '.join(all_vars)}")
        
        # For automation, assume keeping common variables
        keep_vars = ['temperature', 'pressure', 'humidity', 'wind_speed', 'precipitation']
        
        # Filter to keep only specified variables
        variables_to_drop = [var for var in all_vars if var not in keep_vars]
        
        if not variables_to_drop:
            print("No variables to drop")
            return False
        
        # Drop unused variables
        reduced_ds = ds.drop_vars(variables_to_drop)
        
        print(f"Dropped {len(variables_to_drop)} variables")
        print(f"Dropped variables: {', '.join(variables_to_drop)}")
        
        # Calculate memory savings
        original_size = ds.nbytes
        reduced_size = reduced_ds.nbytes
        savings = original_size - reduced_size
        savings_percent = (savings / original_size) * 100
        
        print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
        print(f"Reduced size: {reduced_size / 1024 / 1024:.2f} MB")
        print(f"Memory savings: {savings / 1024 / 1024:.2f} MB ({savings_percent:.1f}%)")
        
        # Write reduced file
        output_file = input_file.replace('.nc', '_optimized.nc')
        reduced_ds.to_netcdf(output_file)
        
        print(f"Optimized file written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def selective_loading(input_file, variables):
    """Load only specified variables to save memory."""
    
    try:
        # Open with selective loading
        ds = xr.open_dataset(input_file, 
                              chunks='auto',
                              drop_variables=variables)
        
        print(f"Opened file with selective loading")
        print(f"Variables dropped: {', '.join(variables)}")
        print(f"Variables kept: {', '.join([v for v in ds.data_vars.keys()]))")
        
        # Calculate memory usage
        memory_usage = ds.nbytes / 1024 / 1024
        print(f"Memory usage: {memory_usage:.2f} MB")
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Optimize memory usage for large datasets')
    parser.add_argument('input_file', help='Input dataset file')
    parser.add_argument('--variable', help='Variable to process')
    parser.add_argument('--operation', choices=['reduce_precision', 'drop_unused', 'selective_loading'], 
                       required=True, help='Memory optimization operation')
    
    args = parser.parse_args()
    
    if args.operation == 'reduce_precision':
        if args.variable is None:
            print("Error: Must specify --variable for reduce_precision operation")
            sys.exit(1)
        success = reduce_precision(args.input_file, args.variable)
    elif args.operation == 'drop_unused':
        success = drop_unused_variables(args.input_file)
    elif args.operation == 'selective_loading':
        if args.variable is None:
            print("Error: Must specify --variable for selective_loading operation")
            sys.exit(1)
        success = selective_loading(args.input_file, [args.variable])
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()