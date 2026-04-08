#!/usr/bin/env python3
"""
Read and display ARM radar data files.

Usage:
    python read_radar.py <radar_file> [--summary] [--fields] [--metadata]
    
Examples:
    python read_radar.py radar.nc
    python read_radar.py radar.nc --summary
    python read_radar.py radar.nc --fields
"""

import sys
import argparse
import pyart


def print_summary(radar):
    """Print summary of radar object."""
    print("\n=== Radar Summary ===")
    print(f"Radar type: {radar.radar_type}")
    print(f"Number of sweeps: {radar.nsweeps}")
    print(f"Number of gates: {radar.ngates}")
    print(f"Number of rays: {radar.nrays}")
    
    if hasattr(radar, 'latitude'):
        print(f"Latitude: {radar.latitude}")
    if hasattr(radar, 'longitude'):
        print(f"Longitude: {radar.longitude}")
    if hasattr(radar, 'altitude'):
        print(f"Altitude: {radar.altitude} m")
    
    print(f"\nAvailable fields: {list(radar.fields.keys())}")


def print_fields(radar):
    """Print information about radar fields."""
    print("\n=== Radar Fields ===")
    
    for field_name, field_dict in radar.fields.items():
        print(f"\n{field_name}:")
        print(f"  Shape: {field_dict['data'].shape}")
        print(f"  Dtype: {field_dict['data'].dtype}")
        print(f"  Missing values: {int(field_dict['data'].mask.sum()) if hasattr(field_dict['data'], 'mask') else 0}")
        
        if 'metadata' in field_dict:
            print(f"  Metadata: {field_dict['metadata']}")


def print_metadata(radar):
    """Print detailed metadata."""
    print("\n=== Radar Metadata ===")
    
    # Radar attributes
    print("\nRadar Attributes:")
    for attr in dir(radar):
        if not attr.startswith('_') and not callable(getattr(radar, attr)):
            try:
                value = getattr(radar, attr)
                if not isinstance(value, dict):
                    print(f"  {attr}: {value}")
            except:
                pass
    
    # Field metadata
    print("\nField Metadata:")
    for field_name, field_dict in radar.fields.items():
        if 'metadata' in field_dict:
            print(f"\n{field_name}:")
            for key, value in field_dict['metadata'].items():
                print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Read ARM radar data')
    parser.add_argument('radar_file', help='Input radar file')
    parser.add_argument('--summary', action='store_true', help='Print radar summary')
    parser.add_argument('--fields', action='store_true', help='Print field information')
    parser.add_argument('--metadata', action='store_true', help='Print detailed metadata')
    
    args = parser.parse_args()
    
    try:
        # Determine file format
        if args.radar_file.endswith('.nc'):
            radar = pyart.io.read_arm_netcdf(args.radar_file)
        elif args.radar_file.endswith('.mdv'):
            radar = pyart.io.read_mdv(args.radar_file)
        elif args.radar_file.endswith('.sigmet'):
            radar = pyart.io.read_sigmet(args.radar_file)
        else:
            print("Unknown file format, trying ARM netCDF...")
            radar = pyart.io.read_arm_netcdfares(args.radar_file)
        
        if not (args.summary or args.fields or args.metadata):
            args.summary = True
            args.fields = True
        
        if args.summary:
            print_summary(radar)
        
        if args.fields:
            print_fields(radar)
        
        if args.metadata:
            print_metadata(radar)
    
    except FileNotFoundError:
        print(f"Error: File not found: {args.radar_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()