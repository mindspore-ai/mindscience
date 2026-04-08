#!/usr/bin/env python3
"""
Apply quality control to radar data.

Usage:
    python quality_control.py <input_radar> <output_radar> [--gate-filter] [--despeckle]
    
Examples:
    python quality_control.py radar.nc radar_qc.nc --gate-filter
    python quality_control.py radar.nc radar_qc.nc --despeckle
"""

import sys
import argparse
import pyart


def apply_gate_filter(input_file, output_file):
    """Apply gate filtering to radar data."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Apply gate filter to reflectivity
        radar_qc = pyart.correct.GateFilter(radar, 
                                            field_name='reflectivity',
                                            min_value=-30,
                                            max_value=70)
        
        # Apply gate filter to velocity
        radar_qc = pyart.correct.GateFilter(radar_qc,
                                            field_name='velocity',
                                            min_value=-30,
                                            max_value=30)
        
        # Apply gate filter to spectrum width
        radar_qc = pyart.correct.GateFilter(radar_qc,
                                            field_name='spectrum_width',
                                            min_value=0,
                                            max_value=10)
        
        # Write output
        pyart.io.write_arm_netcdf(radar_qc, output_file)
        
        print(f"Gate filtering applied successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def apply_despeckle(input_file, output_file):
    """Apply despeckle to radar data."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Apply despeckle to reflectivity
        radar_ds = pyart.correct.despeckle.despeckle_field(
            radar, field_name='reflectivity', 
            threshold=3, 
            gatefilter=False, 
            fsize=5)
        
        # Write output
        pyart.io.write_arm_netcdf(radar_ds, output_file)
        
        print(f"Despeckle applied successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def apply_moment_gate_filter(input_file, output_file):
    """Apply moment and gate filtering."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Apply moment and gate filtering
        radar_qc = pyart.correct.moment_and_gatefilter.GateFilter(
            radar, field_name='reflectivity')
        
        # Write output
        pyart.io.write_arm_netcdf(radar_qc, output_file)
        
        print(f"Moment and gate filtering applied successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Apply quality control to radar data')
    parser.add_argument('input_radar', help='Input radar file')
    parser.add_argument('output_radar', help='Output radar file')
    parser.add_argument('--gate-filter', action='store_true', 
                       help='Apply gate filtering')
    parser.add_argument('--despeckle', action='store_true', 
                       help='Apply despeckle')
    parser.add_argument('--moment-filter', action='store_true', 
                       help='Apply moment and gate filtering')
    
    args = parser.parse_args()
    
    if not (args.gate_filter or args.despeckle or args.moment_filter):
        args.gate_filter = True
        args.moment_filter = True
    
    success = False
    
    if args.gate_filter:
        success = apply_gate_filter(args.input_radar, args.output_radar)
    
    if args.despeckle:
'        success = apply_despeckle(args.input_radar, args.output_radar)
    
    if args.moment_filter:
        success = apply_moment_gate_filter(args.input_radar, args.output_radar)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()