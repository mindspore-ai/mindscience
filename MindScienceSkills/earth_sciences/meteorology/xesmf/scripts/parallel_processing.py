#!/usr/bin/env python3
"""
Process data in parallel.

Usage:
    python parallel_processing.py <input_file> --operation <operation> --workers <num>
    
Examples:
    python parallel_processing.py large_file.nc --operation mean --workers 4
"""

import sys
import argparse
import xarray as xr
from concurrent.futures import ThreadPoolExecutor


def parallel_mean(input_file, num_workers):
    """Calculate mean in parallel."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks='auto')
        
        # Get variable names
        variables = list(ds.data_vars.keys())
        
        # Process each variable in parallel
        def process_var(var_name):
            return ds[var_name].mean().load()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_var, variables))
        
        print(f"Processed {len(results)} variables in parallel")
        for var_name, result in zip(variables, results):
            print(f"  {var_name}: {result:.6f}")
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def parallel_std(input_file, num_workers):
    """Calculate std in parallel."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks='auto')
        
        # Get variable names
        variables = list(ds.data_vars.keys())
        
        # Process each variable in parallel
        def process_var(var_name):
            return ds[var_name].std().load()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_var, variables))
        
        print(f"Processed {len(results)} variables in parallel")
        for var_name, result in zip(variables, results):
            print(f"  {var_name}: {result:.6f}")
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def parallel_minmax(input_file, num_workers):
    """Calculate min/max in parallel."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks='auto')
        
        # Get variable names
        variables = list(ds.data_vars.keys())
        
        # Process each variable in parallel
        def process_var(var_name):
            var = ds[var_name]
            return {
                'min': var.min().load(),
                'max': var.max().load()
            }
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_var, variables))
        
        print(f"Processed {len(results)} variables in parallel")
        for var_name, result in zip(variables, results):
            print(f"  {var_name}: Min={result['min']:.6f}, Max={result['max']:.6f}")
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Process data in parallel')
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('--operation', choices=['mean', 'std', 'minmax'], 
                       required=True, help='Operation to perform')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    success = False
    
    if args.operation == 'mean':
        success = parallel_mean(args.input_file, args.workers)
    elif args.operation == 'std':
        success = parallel_std(args.input_file, args.workers)
    elif args.operation == 'minmax':
        success = parallel_minmax(args.input_file, args.workers)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()