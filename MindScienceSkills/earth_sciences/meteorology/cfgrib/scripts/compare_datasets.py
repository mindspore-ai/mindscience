#!/usr/bin/env python3
"""
Compare two GRIB datasets.

Usage:
    python compare_datasets.py <file1.grib> <file2.grib> [--variable <var>]
    
Examples:
    python compare_datasets.py file1.grib file2.grib
    python compare_datasets.py file1.grib file2.grib --variable t2m
"""

import sys
import argparse
import xarray as xr
import numpy as np


def compare_datasets(file1, file2, variable=None):
    """Compare two GRIB datasets."""
    
    try:
        with xr.open_dataset(file1, engine='cfgrib') as ds1, \
             xr.open_dataset(file2, engine='cfgrib') as ds2:
            
            print("\n=== Dataset Comparison ===")
            print(f"\nFile 1: {file1}")
            print(f"  Dimensions: {dict(ds1.dims)}")
            print(f"  Variables: {list(ds1.data_vars.keys())}")
            
            print(f"\nFile 2: {file2}")
            print(f"  Dimensions: {dict(ds2.dims)}")
            print(f"  Variables: {list(ds2.data_vars.keys())}")
            
            # Compare dimensions
            print("\n=== Dimension Comparison ===")
            for dim in set(list(ds1.dims.keys()) + list(ds2.dims.keys())):
                size1 = ds1.dims.get(dim, None)
                size2 = ds2.dims.get(dim, None)
                
                if size1 == size2:
                    print(f"  {dim}: {size1} (match)")
                else:
                    print(f"  {dim}: {size1} vs {size2} (mismatch)")
            
            # Compare variables
            print("\n=== Variable Comparison ===")
            
            if variable:
                vars_to_compare = [variable]
            else:
                vars_to_compare = set(list(ds1.data_vars.keys()) + list(ds2.data_vars.keys()))
            
            for var in vars_to_compare:
                if var in ds1.data_vars and var in ds2.data_vars:
                    print(f"\n{var}:")
                    
                    data1 = ds1[var]
                    data2 = ds2[var]
                    
                    # Compare shapes
                    if data1.shape == data2.shape:
                        print(f"  Shape: {data1.shape} (match)")
                    else:
                        print(f"  Shape: {data1.shape} vs {data2.shape} (mismatch)")
                        continue
                    
                    # Compare values
                    diff = data1 - data2
                    max_diff = float(np.abs(diff).max())
                    mean_diff = float(np.abs(diff).mean())
                    
                    print(f"  Max absolute difference: {max_diff:.6e}")
                    print(f"  Mean absolute difference: {mean_diff:.6e}")
                    
                    if max_diff < 1e-10:
                        print(f"  Status: IDENTICAL")
                    elif max_diff < 1e-5:
                        print(f"  Status: VERY SIMILAR")
                    elif max_diff < 1e-2:
                        print(f"  Status: SIMILAR")
                    else:
                        print(f"  Status: DIFFERENT")
                
                elif var in ds1.data_vars:
                    print(f"\n{var}: Present in file 1 only")
                elif var in ds2.data_vars:
                    print(f"\n{var}: Present in file 2 only")
            
            return True
    
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Compare two GRIB datasets')
    parser.add_argument('file1', help='First GRIB file')
    parser.add_argument('file2', help='Second GRIB file')
    parser.add_argument('--variable', help='Variable to compare')
    
    args = parser.parse_args()
    
    success = compare_datasets(args.file1, args.file2, args.variable)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()