#!/usr/bin/env python3
"""
Validate GRIB file structure and contents.

Usage:
    python validate_grib.py <input.grib> [--detailed]
    
Examples:
    python validate_grib.py data.grib
    python validate_grib.py data.grib --detailed
"""

import sys
import argparse
import xarray as xr
import numpy as np


def validate_grib_file(input_file, detailed=False):
    """Validate GRIB file structure and contents."""
    
    try:
        with xr.open_dataset(input_file, engine='cfgrib') as ds:
            print("\n=== File Validation ===")
            print(f"File: {input_file}")
            print(f"Dimensions: {dict(ds.dims)}")
            print(f"Variables: {list(ds.data_vars.keys())}")
            print(f"Coordinates: {list(ds.coords.keys())}")
            
            errors = []
            warnings = []
            
            # Check dimensions
            if not ds.dims:
                errors.append("No dimensions found")
            
            # Check variables
            if not ds.data_vars:
                errors.append("No data variables found")
            
            # Check coordinates
            if not ds.coords:
                warnings.append("No coordinates found")
            
            # Check for time coordinate
            if 'time' not in ds.coords:
                warnings.append("No time coordinate found")
            else:
                time = ds.coords['time']
                if len(time) > 1:
                    time_diff = np.diff(time.values)
                    if not np.all(time_diff == time_diff[0]):
                        warnings.append("Time steps are not uniform")
            
            # Check for spatial coordinates
            if 'latitude' not in ds.coords:
                warnings.append("No latitude coordinate found")
            if 'longitude' not in ds.coords:
                warnings.append("No longitude coordinate found")
            
            # Check data values
            if detailed:
                print("\n=== Data Validation ===")
                
                for var_name, var in ds.data_vars.items():
                    print(f"\n{var_name}:")
                    print(f"  Shape: {var.shape}")
                    print(f"  Dtype: {var.dtype}")
                    print(f"  Size: {var.size}")
                    
                    # Check for missing values
                    try:
                        missing_count = int(np.isnan(var).sum())
                        if missing_count > 0:
                            warnings.append(f"{var_name}: {missing_count} missing values")
                            print(f"  Missing values: {missing_count}")
                    except:
                        pass
                    
                    # Check value range
                    try:
                        min_val = float(var.min())
                        max_val = float(var.max())
                        print(f"  Min: {min_val:.6e}")
                        print(f"  Max: {max_val:.6e}")
                        
                        # Check for suspicious values
                        if min_val < -1e30:
                            warnings.append(f"{var_name}: Very small minimum value")
                        if max_val > 1e30:
                            warnings.append(f"{var_name}: Very large maximum value")
                    except:
                        pass
            
            # Print results
            print("\n=== Validation Results ===")
            
            if errors:
                print(f"\n❌ Errors found: {len(errors)}")
                for error in errors:
                    print(f"  - {error}")
            else:
                print("\n✅ No errors found")
            
            if warnings:
                print(f"\n⚠️  Warnings: {len(warnings)}")
                for warning in warnings:
                    print(f"  - {warning}")
            else:
                print("\n✅ No warnings")
            
            return len(errors) == 0
    
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate GRIB file')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('--detailed', action='store_true', 
                       help='Show detailed validation information')
    
    args = parser.parse_args()
    
    success = validate_grib_file(args.input_file, args.detailed)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()