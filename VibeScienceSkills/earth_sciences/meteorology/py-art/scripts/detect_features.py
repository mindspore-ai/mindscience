#!/usr/bin/env python3
"""
Detect radar features (hail, melting layer, etc.).

Usage:
    python detect_features.py <input_radar> <output_radar> [--hail] [--melting-layer] [--turbulence]
    
Examples:
    python detect_features.py radar.nc radar_features.nc --hail
    python detect_features.py radar.nc radar_features.nc --melting-layer
"""

import sys
import argparse
import pyart


def detect_hail(input_file, output_file):
    """Detect hail using radar data."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Check for required fields
        if 'reflectivity' not in radar.fields:
            print("Error: Reflectivity field not found")
            return False
        
        # Detect hail
        radar_hail = pyart.retrieve.detect_hail(radar)
        
        # Write output
        pyart.io.write_arm_netcdf(radar_hail, output_file)
        
        print("Hail detection completed successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def detect_melting_layer(input_file, output_file):
    """Detect melting layer using radar data."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Check for required fields
        if 'reflectivity' not in radar.fields:
            print("Error: Reflectivity field not found")
            return False
        
        # Detect melting layer
        radar_melt = pyart.retrieve.melting_layer(
            radar, 
            field_name='reflectivity',
            threshold=-5.0,
            min_beam_width=2.0
        )
        
        # Write output
        pyart.io.write_arm_netcdf(radar_melt, output_file)
        
        print("Melting layer detection completed successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def detect_turbulence(input_file, output_file):
    """Detect turbulence using radar data."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Check for required fields
        if 'velocity' not in radar.fields:
            print("Error: Velocity field not found")
            return False
        
        # Detect turbulence (simplified)
        # In practice, use more sophisticated algorithms
        radar_turb = pyart.retrieve.detect_turbulence(radar)
        
        # Write output
        pyart.io.write_arm_netcdf(radar_turb, output_file)
        
        print("Turbulence detection completed successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Detect radar features')
    parser.add_argument('input_radar', help='Input radar file')
    parser.add_argument('output_radar', help='Output radar file')
    parser.add_argument('--hail', action='store_true', help='Detect hail')
    parser.add_argument('--melting-layer', action='store_true', help='Detect melting layer')
    parser.add_argument('--turbulence', action='store_true', help='Detect turbulence')
    
    args = parser.parse_args()
    
    if not (args.hail or args.melting_layer or args.turbulence):
        print("Error: Must specify at least one detection type")
        sys.exit(1)
    
    success = False
    
    if args.hail:
        success = detect_hail(args.input_radar, args.output_radar)
    
    if args.melting_layer:
        success = detect_melting_layer(args.input_radar, args.output_radar)
    
    if args.turbulence:
        success = detect_turbulence(args.input_radar, args.output_radar)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()