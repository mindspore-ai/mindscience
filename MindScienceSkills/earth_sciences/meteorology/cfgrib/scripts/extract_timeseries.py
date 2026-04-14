#!/usr/bin/env python3
"""
Extract time series for specific locations from GRIB files.

Usage:
    python extract_timeseries.py <input.grib> <output.csv> --lat <lat> --lon <lon>
    python extract_timeseries.py <input.grib> <output.csv> --locations <lat1,lon1,lat2,lon2,...>
    
Examples:
    python extract_timeseries.py data.grib output.csv --lat 40.0 --lon -100.0
    python extract_timeseries.py data.grib output.csv --locations 40.0,-100.0,35.0,-105.0
"""

import sys
import argparse
import xarray as xr
import pandas as pd


def extract_timeseries(input_file, output_file, locations, variable='t2m'):
    """Extract time series for specific locations."""
    
    try:
        with xr.open_dataset(input_file, engine='cfgrib') as ds:
            if variable not in ds.data_vars:
                print(f"Error: Variable '{variable}' not found in dataset")
                print(f"Available variables: {list(ds.data_vars.keys())}")
                return False
            
            # Parse locations
            if len(locations) == 2:
                # Single location
                lats = [locations[0]]
                lons = [locations[1]]
            elif len(locations) % 2 == 0:
                # Multiple locations
                lats = locations[::2]
                lons = locations[1::2]
            else:
                print("Error: Invalid number of coordinates")
                return False
            
            # Extract time series for each location
            all_series = []
            
            for lat, lon in zip(lats, lons):
                # Select nearest point
                ts = ds[variable].sel(latitude=lat, longitude=lon, method='nearest')
                
                # Convert to pandas Series
                ts_series = ts.to_series()
                ts_series.name = f"{variable}_lat{lat}_lon{lon}"
                
                all_series.append(ts_series)
                
                print(f"Extracted time series for ({lat}, {lon})")
            
            # Combine into DataFrame
            df = pd.concat(all_series, axis=1)
            
            # Write to CSV
            df.to_csv(output_file)
            
            print(f"\nTime series written to: {output_file}")
            print(f"Shape: {df.shape}")
            print(f"Time range: {df.index[0]} to {df.index[-1]}")
            
            return True
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Extract time series from GRIB files')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('output_file', help='Output CSV file')
    parser.add_argument('--lat', type=float, help='Latitude')
    parser.add_argument('--lon', type=float, help='Longitude')
    parser.add_argument('--locations', help='Comma-separated lat,lon pairs')
    parser.add_argument('--variable', default='t2m', help='Variable name (default: t2m)')
    
    args = parser.parse_args()
    
    if args.lat and args.lon:
        locations = [args.lat, args.lon]
    elif args.locations:
        try:
            locations = [float(x) for x in args.locations.split(',')]
        except ValueError:
            print("Error: Invalid coordinates format")
            sys.exit(1)
    else:
        print("Error: Must specify either --lat/--lon or --locations")
        sys.exit(1)
    
    success = extract_timeseries(args.input_file, args.output_file, locations, args.variable)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()