#!/usr/bin/env python3
"""
Export radar data to various formats.

Usage:
    python export_data.py <radar_file> <output_file> --format <format>
    
Examples:
    python export_data.py radar.nc output.nc --format netcdf
    python export_data.py radar.nc output.csv --format csv
    python export_data.py radar.nc output.png --format png
"""

import sys
import argparse
import pyart
import numpy as np


def export_to_netcdf(input_file, output_file):
    """Export radar data to NetCDF format."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Write to NetCDF
        pyart.io.write_arm_netcdf(radar, output_file)
        
        print(f"Radar data exported to NetCDF: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def export_to_csv(input_file, output_file, field_name='reflectivity'):
    """Export radar data to CSV format."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Check if field exists
        if field_name not in radar.fields:
            print(f"Error: Field '{field_name}' not found")
            print(f"Available fields: {list(radar.fields.keys())}")
            return False
        
        # Get field data
        field_data = radar.fields[field_name]['data']
        
        # Flatten data
        flattened = field_data.flatten()
        
        # Create CSV with metadata
        import pandas as pd
        
        # Create DataFrame with coordinates
        df = pd.DataFrame({
            'sweep': np.repeat(np.arange(radar.nsweeps), radar.ngates),
            'gate': np.tile(np.arange(radar.ngates), radar.nsweeps),
            field_name: flattened
        })
        
        # Write to CSV
        df.to_csv(output_file, index=False)
        
        print(f"Radar data exported to CSV: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def export_to_png(input_file, output_file, field_name='reflectivity'):
    """Export radar data as PNG image."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Check if field exists
        if field_name not in radar.fields:
            print(f"Error: Field '{field_name}' not found")
            return False
        
        # Create display
        display = pyart.graph.RadarDisplay(radar)
        
        # Set color limits
        if field_name == 'reflectivity':
            vmin, vmax = -30, 70
            cmap = 'NWSpectral'
        else:
            vmin, vmax = None, None
            cmap = 'jet'
        
        # Create figure
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
        # Plot field
        display.plot(field_name, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)
        
        # Add colorbar
        plt.colorbar(ax.collections[0], ax=ax, label=field_name)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Radar data exported to PNG: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export radar data to various formats')
    parser.add_argument('radar_file', help='Input radar file')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('--format', choices=['netcdf', 'csv', 'png'], 
                       required=True, help='Output format')
    parser.add_argument('--field', default='reflectivity', help='Field name for CSV/PNG export')
    
    args = parser.parse_args()
    
    success = False
    
    if args.format == 'netcdf':
        success = export_to_netcdf(args.radar_file, args.output_file)
    elif args.format == 'csv':
        success = export_to_csv(args.radar_file, args.output_file, args.field)
    elif args.format == 'png':
        success = export_to_png(args.radar_file, args.output_file, args.field)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()