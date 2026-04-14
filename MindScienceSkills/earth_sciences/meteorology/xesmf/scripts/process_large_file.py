#!/usr/bin/env python3
"""
Process large GRIB/NetCDF files in chunks.

Usage:
    python process_large_file.py <input_file> --output <output_file> --time-chunk <size> --lat-chunk <size> --lon-chunk <size>
    
Examples:
    python process_large_file.py large_file.nc output.nc --time-chunk 10 --lat-chunk 90 --lon-chunk 180
"""

import sys
import argparse
import xarray as xr
import numpy as np


def process_in_chunks(input_file, output_file, time_chunk, lat_chunk, lon_chunk):
    """Process large file in chunks."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks={
            'time': time_chunk,
            'latitude': lat_chunk,
            'longitude': lon_chunk
        })
        
        print(f"Processing file: {input_file}")
        print(f"Chunks: {ds.chunks}")
        print(f"Dimensions: {dict(ds.dims)}")
        
        # Calculate statistics in chunks
        if 'temperature' in ds.data_vars:
            print("\nProcessing temperature field...")
            
            # Process time chunks
            for i in range(0, len(ds.time), time_chunk):
                time_chunk_data = ds.isel(time=slice(i, i+time_chunk))
                
                # Calculate spatial mean for this time chunk
                spatial_mean = time_chunk_data['temperature'].mean(dim=['latitude', 'longitude'])
                
                print(f"  Time chunk {i//time_chunk + 1}: "
                      f"Mean: {spatial_mean:.2f} K")
                
                # Calculate statistics
                chunk_mean = time_chunk_data['temperature'].mean()
                chunk_std = time_chunk_data['temperature'].std()
                chunk_min = time_chunk_data['temperature'].min()
                chunk_max = time_chunk_data['maximum'].max()
                
                print(f"    Mean: {chunk_mean:.2f} K")
                print(f"    Std: {chunk_std:.2f} K")
                print(f"    Min: {chunk_min:.2f} K")
                print(f"    Max: {chunk.max:.2f} K")
            
            # Calculate overall statistics
            overall_mean = ds['temperature'].mean()
            overall_std = ds['temperature'].std()
            overall_min = ds['temperature'].min()
            overall_max = ds['temperature'].max()
            
            print(f"\nOverall statistics:")
            print(f"  Mean: {overall_mean:.2f} K")
            print(f"  Std: {overall_std:.2f} K")
            print(f"  Min: {overall_min:.2f} K")
            print(f"  Max: {overall_max:.2f} K")
        
        # Write output if specified
        if output_file:
            ds.to_netcdf(output_file)
            print(f"\nOutput written to: {output_file}")
        
        return True
    
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Process large GRIB/NetCDF files in chunks')
    parser.add_argument('input_file', help='Input GRIB/NetCDF file')
    parser.add_argument('--output', help='Output NetCDF file')
    parser.add_argument('--time-chunk', type=int, default=10, help='Time chunk size')
    parser.add_argument('--lat-chunk', type=int, default=90, help='Latitude chunk size')
    parser.add_argument('--lon-chunk', type=int, default=180, help='Longitude chunk size')
    
    args = parser.parse_args()
    
    success = process_in_chunks(args.input_file, args.output, 
                                args.time_chunk, args.lat_chunk, args.lon_chunk)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()