#!/usr/bin/env python3
"""
Analyze data in chunks.

Usage:
    python chunk_analysis.py <input_file> --field <field> --time-chunk <size> --lat-chunk <size> --lon-chunk <size>
    
Examples:
    python chunk_analysis.py large_file.nc --field temperature --time-chunk 10 --lat-chunk 90 --lon-chunk 180
"""

import sys
import argparse
import xarray as xr
import numpy as np


def analyze_in_chunks(input_file, field_name, time_chunk, lat_chunk, lon_chunk):
    """Analyze data in chunks."""
    
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
        
        print(f"Analyzing field: {field_name}")
        print(f"Chunks: {field.chunks}")
        print(f"Total chunks: {len(field.chunks)}")
        
        # Analyze each chunk
        chunk_stats = []
        
        for i in range(len(field.time) // time_chunk):
            # Get chunk
            time_slice = field.isel(time=slice(i*time_chunk, (i+1)*time_chunk))
            
            # Calculate statistics
            chunk_mean = time_slice.mean().load()
            chunk_std = time_slice.std().load()
            chunk_min = time_slice.min().load()
            chunk_max = time_slice.max().load()
            
            chunk_stats.append({
                'chunk': i+1,
                'mean': float(chunk_mean),
                'std': float(chunk_std),
                'min': float(chunk_min),
                'max': float(chunk_max)
            })
            
            print(f"  Chunk {i+1}: "
                  f"Mean={chunk_mean:.2f}, "
                  f"Std={chunk_std:.2f}, "
                  f"Min={chunk_min:.2f}, "
                  f"Max={chunk_max:.2f}")
        
        # Calculate overall statistics
        overall_mean = field.mean().load()
        overall_std = field.std().load()
        overall_min = field.min().load()
        overall_max = field.max().load()
        
        print(f"\nOverall statistics:")
        print(f"  Mean: {overall_mean:.2f}")
        print(f"  Std: {overall_std:.2f}")
        print(f"  Min: {overall_min:.2f}")
        print(f"  Max: {overall_max:.2f}")
        
        return True
    
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Analyze data in chunks')
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('--field', required=True, help='Field name to analyze')
    parser.add_argument('--time-chunk', type=int, default=10, help='Time chunk size')
    parser.add_argument('--lat-chunk', type=int, default=90, help='Latitude chunk size')
    parser.add_argument('--lon-chunk', type=int, default=180, help='Longitude chunk size')
    
    args = parser.parse_args()
    
    success = analyze_in_chunks(args.input_file, args.field, 
                                args.time_chunk, args.lat_chunk, args.lon_chunk)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()