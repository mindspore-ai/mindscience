#!/usr/bin/env python3
"""
Convert large files to other formats with chunking.

Usage:
    python convert_format.py <input_file> <output_file> --format <format> --time-chunk <size> --lat-chunk <size> --lon-chunk <size>
    
Examples:
    python convert_format.py large_file.nc output.nc --format netcdf --time-chunk 10.0
    python convert_format.py large_file.nc output.zarr --format zarr --time-chunk 10.0 --lat-chunk 90.0 --lon-chunk 180.0
"""

import sys
import argparse
import xarray as xr


def convert_to_netcdf(input_file, output_file, chunks):
    """Convert to NetCDF with chunking."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks=chunks)
        
        # Create encoding with compression
        encoding = {
            var: {
                'zlib': True, 
                'complevel': 5,
                'chunksizes': chunks
            } for var in ds.data_vars
        }
        
        # Write to NetCDF
        ds.to_netcdf(output_file, encoding=encoding)
        
        print(f"Conversion to NetCDF completed")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Chunks: {chunks}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def convert_to_zarr(input_file, output_file, chunks):
    """Convert to Zarr with chunking."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks=chunks)
        
        # Create encoding with chunking
        encoding = {
            var: {
                'chunksizes': chunks
            } for var in ds.data_vars
        }
        
        # Write to Zarr
        ds.to_zarr(output_file, encoding=encoding)
        
        print(f"Conversion to Zarr completed")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Chunks: {chunks}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def convert_to_csv(input_file, output_file, field_name, chunks):
    """Convert to CSV with chunking."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks=chunks)
        
        # Check if field exists
        if field_name not in ds.data_vars:
            print(f"Error: Field '{field_name}' not found")
            print(f"Available fields: {list(ds.data_vars.keys())}")
            return False
        
        # Get field and convert to pandas
        field = ds[field_name]
        
        # Convert to CSV in chunks
        for i in range(0, len(field.time), chunks['time']):
            time_chunk = field.isel(time=slice(i, i + chunks['time']))
            
            # Convert chunk to DataFrame
            df = time_chunk.to_dataframe()
            
            # Write chunk to CSV
            mode = 'a' if i == 0 else 'a'
            df.to_csv(output_file, mode=mode, header=(i == 0))
            
            print(f"Processed time chunk {i//chunks['time'] + 1}")
        
        print(f"Conversion to CSV completed")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Field: {field_name}")
        print(f"Chunks: {chunks}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert large files to other formats')
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('--format', choices=['netcdf', 'zarr', 'csv'], 
                       required=True, help='Output format')
    parser.add_argument('--field', help='Field name for CSV conversion')
    parser.add_argument('--time-chunk', type=int, default=10, help='Time chunk size')
    parser.add_argument('--lat-chunk', type=int, default=90, help='Latitude chunk size')
    parser.add_argument('--lon-chunk', type=int, default=180, help='Longitude chunk size')
    
    args = parser.parse_args()
    
    # Define chunks
    chunks = {
        'time': args.time_chunk,
        'latitude': args.lat_chunk,
        'longitude': args.lon_chunk
    }
    
    success = False
    
    if args.format == 'netcdf':
        success = convert_to_netcdf(args.input_file, args.output_file, chunks)
    elif args.format == 'zarr':
        success = convert_to_zarr(args.input_file, args.output_file, chunks)
    elif args.format == 'csv':
        if args.field is None:
            print("Error: Must specify --field for CSV conversion")
            sys.exit(1)
        success = convert_to_csv(args.input_file, args.output_file, args.field, chunks)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()