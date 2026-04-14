#!/usr/bin/env python3
"""
Analyze polarimetric radar data.

Usage:
    python polarimetric_analysis.py <input_radar> <output_radar> [--hydro-class] [--melting-layer]
    
Examples:
    python polarimetric_analysis.py radar.nc radar_hydro.nc --hydro-class
    python polarimetric_analysis.py radar.nc radar_melt.nc --melting-layer
"""

import sys
import argparse
import pyart


def classify_hydrometeors(input_file, output_file):
    """Classify hydrometeors using polarimetric data."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Check for required fields
        if 'reflectivity' not in radar.fields:
            print("Error: Reflectivity field not found")
            return False
        
        # Classify hydrometeors
        radar_hydro = pyart.retrieve.hydroclass_hs(
            radar, 
            field_name='reflectivity',
            hydro_class='HS'
        )
        
        # Write output
        pyart.io.write_arm_netcdf(radar_hydro, output_file)
        
        print("Hydrometeor classification completed successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def detect_melting_layer(input_file, output_file):
    """Detect melting layer using reflectivity data."""
    
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


def compute_polarimetric_variables(input_file, output_file):
    """Compute polarimetric variables."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Compute ZH
        if 'reflectivity' in radar.fields:
            radar = pyart.retrieve.compute_zh(radar)
            print("ZH computed")
        
        # Compute ZDR
        if 'differential_reflectivity' in radar.fields:
            radar = pyart.retrieve.compute_zdr(radar)
            print("ZDR computed")
        
        # Compute KDP
        if 'differential_phase' in radar.fields:
            radar = pyart.retrieve.compute_kdp(radar)
            print("KDP computed")
        
        # Write output
        pyart.io.write_arm_netcdf(radar, output_file)
        
        print("Polarimetric variables computed successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Analyze polarimetric radar data')
    parser.add_argument('input_radar', help='Input radar file')
    parser.add_argument('output_radar', help='Output radar file')
    parser.add_argument('--hydro-class', action='store_true', 
                       help='Classify hydrometeors')
    parser.add_argument('--melting-layer', action='store_true', 
                       help='Detect melting layer')
    parser.add_argument('--polarimetric-vars', action='store_true', 
                       help='Compute polarimetric variables')
    
    args = parser.parse_args()
    
    if not (args.hydro_class or args.melting_layer or args.polarimetric_vars):
        print("Error: Must specify at least one analysis type")
        sys.exit(1)
    
    success = False
    
    if args.hydro_class:
        success = classify_hydrometeors(args.input_radar, args.output_radar)
    
    if args.melting_layer:
        success = detect_melting_layer(args.input_radar, args.output_radar)
    
    if args.polarimetric_vars:
        success = compute_polarimetric_variables(args.input_radar, args.output_radar)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()