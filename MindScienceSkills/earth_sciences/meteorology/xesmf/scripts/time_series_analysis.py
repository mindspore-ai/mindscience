#!/usr/bin/env python3
"""
Time series analysis on large datasets.

Usage:
    python time_series_analysis.py <input_file> --location <lat> <lon> --output <output.csv>
    
Examples:
    python time_series_analysis.py large_file.nc --location 40.0 -100.0 --output timeseries.csv
"""

import sys
import argparse
import xarray as xr
import numpy as np


def extract_timeseries(input_file, lat, lon, output_file):
    """Extract time series for a location from large dataset."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks={'time': 10})
        
        # Check for required fields
        if 'time' not in ds.coords:
'            print("Error: Dataset does not have time dimension")
            return False
        
        # Check for spatial coordinates
        if 'latitude' not in ds.coords or 'longitude' not in ds.coords:
            print("Error: Dataset does not have spatial coordinates")
            return False
        
        # Select location
        ts = ds['temperature'].sel(latitude=lat, longitude=lon, method='nearest')
        
        # Convert to pandas Series
        ts_series = ts.to_series()
        
        # Calculate statistics
        mean = ts_series.mean()
        std = ts_series.std()
        min_val = ts_series.min()
        max_val = ts_series.max()
        
        # Calculate trend (linear regression)
        time_numeric = (ts_series.index - ts_series.index[0]).total_seconds() / 86400.0  # Convert to days
        if len(time_numeric) > 1:
            coeffs = np.polyfit(time_numeric, ts_series.values, 1)
            trend = coeffs[0]  # K/day
        else:
            trend = 0.0
        
        # Write to CSV
        ts_series.to_csv(output_file)
        
        print(f"Time series extracted successfully")
        print(f"Time range: {ts_series.index[0]} to {ts_series.index[-1]}")
        print(f"Number of points: {len(ts_series)}")
        print(f"Mean: {mean:.2f}")
        print(f"Std: {std:.2f}")
        print(f"Range: {min_val:.2f} - {max_val:.2f}")
        print(f"Trend: {trend:.6f} K/day")
        print(f"Output written to: {output_file}")
        return True
    
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def extract_regional_timeseries(input_file, lat_north, lat_south, lon_west, lon_east, output_file):
    """Extract regional average time series."""
    
    try:
        # Open with chunking
        ds = xr.open_dataset(input_file, chunks={'time': 10})
        
        # Check for required fields
        if 'temperature' not in ds.data_vars:
            print("Error: Temperature field not found")
            return False
        
        # Select region
        region = ds['temperature'].sel(
            latitude=slice(lat_north, lat_south),
            longitude=slice(lon_west, lon_east)
        )
        
        # Calculate regional mean time series
        regional_ts = region.mean(dim=['latitude', 'longitude'])
        
        # Convert to pandas Series
        ts_series = regional_ts.to_series()
        
        # Write to CSV
        ts_series.to_csv(output_file)
        
        print(f"Regional time series extracted successfully")
        print(f"Time range: {ts_series.index[0]} to {ts_series.index[-1]}")
        print(f"Number of points: {len(ts_series)}")
        print(f"Output written to: {output_file}")
        return True
    
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Time series analysis on large datasets')
    parser.add_argument('input_file', help='Input dataset file')
    parser.add_argument('--location', nargs=2, type=float, 
                       help='Location (lat lon) for point time series')
    parser.add_argument('--region', nargs=4, type=float,
                       help='Region (lat_north lat_south lon_west lon_east)')
    parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    success = False
    
    if args.location:
        success = extract_timeseries(args.input_file, args.location[0], args.location[1], args.output)
    elif args.region:
        success = extract_regional_timeseries(args.input_file, 
                                               args.region[0], args.region[1], 
                                               args.region[2], args.region[3], 
                                               args.output)
    else:
        print("Error: Must specify either --location or --region")
        sys.exit(1)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()