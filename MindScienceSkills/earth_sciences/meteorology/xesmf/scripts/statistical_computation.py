#!/usr/bin/env python3
"""
Calculate statistics on large datasets.

Usage:
    python statistical_computation.py <input_file> --field <field> --operation <operation>
    
Examples:
    python statistical_computation.py large_file.nc --field temperature --operation mean
    python statistical_computation.py large_file.nc --field temperature --operation std
"""

import sys
import argparse
import xarray as xr
import numpy as np


def calculate_statistics(input_file, field_name, operation):
    """Calculate statistics on large dataset."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks='auto')
        
        # Check if field exists
        if field_name not in ds.data_vars:
            print(f"Error: Field '{field_name}' not found")
            print(f"Available fields: {list(ds.data_vars.keys())}")
            return False
        
        field = ds[field_name]
        
        print(f"Calculating {operation} for {field_name}...")
        
        if operation == 'mean':
            result = field.mean().load()
            print(f"Mean: {result:.6f}")
        
        elif operation == 'std':
            result = field.std().load()
            print(f"Std: {result:.6f}")
        
        elif operation == 'min':
            result = field.min().load()
            print(f"Min: {result:.6f}")
        
        elif operation == 'max':
            result = field.max().load()
            print(f"Max: {result:.6f}")
        
        elif operation == 'median':
            result = field.median().load()
            print(f"Median: {result:.6f}")
        
        elif operation == 'percentiles':
            p25 = field.quantile(0.25).load()
            p50 = field.quantile(0.50).load()
            p75 = field.quantile(0.75).load()
            p90 = field.quantile(0.90).load()
            
            print(f"25th percentile: {p25:.6f}")
            print(f"50th percentile: {p50:.6f}")
            print(f"75th percentile: {p75:.6f}")
            print(f"90th percentile: {p90:.6f}")
        
        elif operation == 'all':
            mean = field.mean().load()
            std = field.std().load()
            min_val = field.min().load()
            max_val = field.max().load()
            median = field.median().load()
            
            print(f"Mean: {mean:.6f}")
            print(f"Std: {std:.6f}")
            print(f"Min: {min_val:.6f}")
            print(f"Max: {max_val:.6f}")
            print(f"Median: {median:.6f}")
        
        else:
            print(f"Error: Unknown operation: {operation}")
            return False
        
        return True
    
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Calculate statistics on large datasets')
    parser.add_argument('input_file', help='Input dataset file')
    parser.add_argument('--field', required=True, help='Field name')
    parser.add_argument('--operation', choices=['mean', 'std', 'min', 'max', 'median', 'percentiles', 'all'], 
                       default='mean', help='Statistical operation')
    
    args = parser.parse_args()
    
    success = calculate_statistics(args.input_file, args.field, args.operation)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()