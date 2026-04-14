#!/usr/bin/env python3
"""
Create visualizations of GRIB data.

Usage:
    python plot_data.py <input.grib> --variable <var> --output <output.png>
    
Examples:
    python plot_data.py data.grib --variable t2m --output plot.png
    python plot_data.py data.grib --variable t2m --time 0 --output plot.png
"""

import sys
import argparse
import xarray as xr
import matplotlib.pyplot as plt


def plot_spatial_field(ds, variable, time_idx=0, output_file=None):
    """Plot spatial field."""
    
    if variable not in ds.data_vars:
        print(f"Error: Variable '{variable}' not found")
        return False
    
    var = ds[variable]
    
    # Select time
    if 'time' in var.dims:
        var = var.isel(time=time_idx)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data
    var.plot(ax=ax, cmap='coolwarm')
    
    # Add title
    plt.title(f'{variable} - Time index {time_idx}')
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return True


def plot_timeseries(ds, variable, lat, lon, output_file=None):
    """Plot time series at a location."""
    
    if variable not in ds.data_vars:
        print(f"Error: Variable '{variable}' not found")
        return False
    
    var = ds[variable]
    
    # Select location
    ts = var.sel(latitude=lat, longitude=lon, method='nearest')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot time series
    ts.plot(ax=ax, marker='o', linestyle='-', markersize=4)
    
    # Add labels
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{variable} ({var.attrs.get("units", "")})')
    ax.set_title(f'{variable} at ({lat}°N, {lon}°E)')
    ax.grid(True)
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description='Visualize GRIB data')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('--variable', required=True, help='Variable name')
    parser.add_argument('--time', type=int, default=0, help='Time index')
    parser.add_argument('--lat', type=float, help='Latitude for time series')
    parser.add_argument('--lon', type=float, 
help='Longitude for time series')
    parser.add_argument('--output', help='Output file name')
    parser.add_argument('--timeseries', action='store_true', 
                       help='Plot time series instead of spatial field')
    
    args = parser.parse_args()
    
    try:
        with xr.open_dataset(args.input_file, engine='cfgrib') as ds:
            if args.timeseries:
                if args.lat is None or args.lon is None:
                    print("Error: Must specify --lat and --lon for time series")
                    sys.exit(1)
                success = plot_timeseries(ds, args.variable, args.lat, args.lon, args.output)
            else:
                success = plot_spatial_field(ds, args.variable, args.time, args.output)
    
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()