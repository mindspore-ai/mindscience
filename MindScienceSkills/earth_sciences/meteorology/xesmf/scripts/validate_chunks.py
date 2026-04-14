#!/usr/bin/env python3
"""
Validate chunked data processing.

Usage:
    python validate_chunks.py <input_file> --field <field> --check-ranges
"""

import sys
import argparse
import xarray as xr
import numpy as np


def validate_chunks(input_file, field_name, check_ranges=True):
    """Validate chunked data processing."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks='auto')
        
        # Check if field exists
        if field_name not in ds.data_vars:
            print(f"Error: Field '{field_name}' not found")
            print(f"Available fields: {list(ds.data_vars.keys())}")
            return False
        
        field = ds[field_name]
        
        print(f"Validating field: {field_name}")
        print(f"Chunks: {field.chunks}")
        print(f"Shape: {field.shape}")
        print(f"Total chunks: {len(field.chunks)}")
        
        # Check for missing data
        if hasattr(field, 'data') and hasattr(field.data, 'mask'):
            total_values = field.size
            missing_count = int(field.data.mask.sum())
            if missing_count > 0:
                missing_percent = (missing_count / total_values) * 100
                print(f"Missing values: {missing_count} ({missing_percent:.1f}%)")
        
        # Check data ranges if requested
        if check_ranges:
            field_data = field.data
            field_clean = field_data[~np.isnan(field_data)]
            
            if len(field_clean) > 0:
                field_min = float(np.min(field_clean))
                field_max = float(np.max(field_clean))
                
                print(f"\nData range: {field_min:.6f} to {field_max:.6f}")
                
                # Check for outliers
                field_mean = float(np.mean(field_clean))
                field_std = float(np.std(field_clean))
                
                outliers = np.abs(field_clean - field_mean) > 3 * field_std
                outlier_count = np.sum(outliers)
                
                if outlier_count > 0:
                    outlier_percent = (outlier_count / len(field_clean)) * 100
                    print(f"Outliers: {outlier_count} ({outlier_percent:.1f}%)")
                
                # Check for unrealistic values
                if field_name == 'reflectivity':
                    if field_min < -30 or field_max > 70:
                        print(f"Warning: Reflectivity values outside expected range")
                    print(f"  Expected: -30 to 70 dBZ")
                elif field_name == 'velocity':
                    if field_min < -30 or field_max > 30:
                        print(f"Warning: Velocity values outside expected range")
                        print(f"  Expected: -30 to 30 m/s")
                elif field_name == 'spectrum_width':
                    if field_min < 0 or field_max > 10:
                        print(f"Warning: Spectrum width values outside expected range")
                        print(f"  Expected: 0 to 10 m/s")
        
        # Check chunk consistency
        chunk_sizes = [chunk.size for chunk in field.chunks]
        if len(set(chunk_sizes)) > 1:
            print(f"Warning: Variable chunk sizes are not consistent")
            print(f"  Chunk sizes: {set(chunk_sizes)}")
        
        print("\n✅ Validation completed successfully")
        return True
    
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate chunked data processing')
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('--field', help='Field name to validate')
    parser.add_argument('--check-ranges', action='store_true', 
                       help='Check data ranges')
    
    args = parser.parse_args()
    
    if args.field is None:
        print("Error: Must specify --field")
        sys.exit(1)
    
    success = validate_chunks(args.input_file, args.field, args.check_ranges)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()