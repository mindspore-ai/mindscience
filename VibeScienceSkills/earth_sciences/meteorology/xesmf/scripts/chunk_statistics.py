#!/usr/bin/env python3
"""
Calculate statistics by chunk.

Usage:
    python chunk_statistics.py <input_file> --field <field> --time-chunk <size> --lat-chunk <size> --lon-chunk <size>
    
Examples:
    python chunk_statistics.py large_file.nc --field temperature --time-chunk 10 --lat-chunk 90 --lon-chunk 180
"""

import sys
import argparse
import xarray as xr
import numpy as np


def calculate_chunk_statistics(input_file, field_name, time_chunk, lat_chunk, lon_chunk):
    """Calculate statistics for each chunk."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks={
            'time': time_chunk,
            'latitude': lat_chunk,
            'longitude': lon_chunk
        })
        
        # Check if field exists
        if field_name not in ds.data_vars:
            print(f"Error: Field '{field_name}' not found")
            print(f"Available fields: {list(ds.data_vars.keys())}")
            return False
        
        # Get field
        field = ds[field_name]
        
        # Calculate statistics for each time chunk
        chunk_stats = []
        
        for i in range(0, len(field.time), time_chunk):
            time_slice = field.isel(time=slice(i, i+time_chunk))
            
            # Calculate statistics
            mean = float(time_slice.mean())
            std = float(time_slice.std())
            min_val = float(time_slice.min())
            max_val = float(time_slice.max())
            
            chunk_stats.append({
                'time_start': str(field.time[i]),
                'time_end': str(field.time[min(i+time_chunk-1)]),
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val
            })
            
            print(f"Time chunk {i//time_chunk + 1}: "
                  f"Mean: {mean:.6f}, "
                  f"Std: {std:.6f}, "
                  f"Range: {min_val:.6f} - {max_val:.6f}")
        
        # Calculate overall statistics
        overall_mean = float(field.mean())
        overall_std = float(field.std())
        overall_min = float(field.min())
        overall_max = float(field.max())
        
        print(f"\nOverall statistics:")
        print(f"  Mean: {overall_mean:.6f}")
        print(f"  Std: {overall_std:.6f}")
        print(f"  Min: {overall_min:.6f}")
        print(f"  Max: {overall_max:.6f}")
        
        return True
    
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Calculate statistics by chunk')
    parser.add_argument('input_file', help='Input dataset file')
    parser.add_argument('--field', required=True, help='Field name')
    parser.add_argument('--time-chunk', type=int, default=10, help='Time chunk size')
    parser.add_argument('--lat-chunk', type=int, default=90, help='Latitude chunk size')
    parser.add_argument('--lon-chunk', type=int, default=180, help='Longitude chunk size')
    
    args = parser.parse_args()
    
    success = calculate_chunk_statistics(args.input_file, args.field, 
                                       args.time_chunk, args.lat_chunk, args.lon_chunk)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()