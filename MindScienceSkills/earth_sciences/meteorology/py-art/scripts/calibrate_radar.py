#!/usr/bin/env python3
"""
Calibrate and correct radar data.

Usage:
    python calibrate_radar.py <input_radar> <output_radar> [--zdr] [--rhohv] [--phidp]
    
Examples:
    python calibrate_radar.py radar.nc radar_cal.nc --zdr
    python calibrate_radar.py radar.nc radar_cal.nc --zdr --rhohv --phidp
"""

import sys
import argparse
import pyart


def calibrate_zdr(input_file, output_file):
    """Calibrate differential reflectivity."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Check for ZDR field
        if 'differential_reflectivity' not in radar.fields:
            print("Warning: ZDR field not found")
            return False
        
        # Calibrate ZDR
        radar_cal = pyart.correct.correct_zdr(radar)
        
        # Write output
        pyart.io.write_arm_netcdf(radar_cal, output_file)
        
        print("ZDR calibration applied successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def correct_rhohv(input_file, output_file):
    """Correct correlation coefficient."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Check for RHOHV field
        if 'cross_correlation_ratio' not in radar.fields:
            print("Warning: RHOHV field not found")
            return False
        
        # Correct RHOHV
        radar_cal = pyart.correct.correct_rhohv(radar)
        
        # Write output
        pyart.io.write_arm_netcdf(radar_cal, output_file)
        
        print("RHOHV correction applied successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def correct_phidp(input_file, output_file):
    """Correct differential phase."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Check for PhiDP field
        if 'differential_phase' not in radar.fields:
            print("Warning: PhiDP field not found")
            return False
        
        # Correct PhiDP
        radar_cal = pyart.correct.correct_phidp(radar)
        
        # Write output
        pyart.io.write_arm_netcdf(radar_cal, output_file)
        
        print("PhiDP correction applied successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def correct_attenuation(input_file, output_file):
    """Correct for attenuation."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Check for reflectivity field
        if 'reflectivity' not in radar.fields:
            print("Warning: Reflectivity field not found")
            return False
        
        # Correct attenuation
        radar_cal = pyart.correct.correct_attenuation_hb(
            radar, field_name='reflectivity')
        
        # Write output
        pyart.io.write_arm_netcdf(radar_cal, output_file)
        
        print("Attenuation correction applied successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Calibrate and correct radar data')
    parser.add_argument('input_radar', help='Input radar file')
    parser.add_argument('output_radar', help='Output radar file')
    parser.add_argument('--zdr', action='store_true', help='Calibrate ZDR')
    parser.add_argument('--rhohv', action='store_true', help='Correct RHOHV')
    parser.add_argument('--phidp', action='store_true', help='Correct PhiDP')
    parser.add_argument('--attenuation', action='store_true', help='Correct attenuation')
    
    args = parser.parse_args()
    
    if not (args.zdr or args.rhohv or args.phidp or args.attenuation):
        print("Error: Must specify at least one correction method")
        sys.exit(1)
    
    success = False
    
    if args.zdr:
        success = calibrate_zdr(args.input_radar, args.output_radar)
    
    if args.rhohv:
        success = correct_rhohv(args.input_radar, args.output_radar)
    
    if args.phidp:
        success = correct_phidp(args.input_radar, args.output_radar)
    
    if args.attenuation:
        success = correct_attenuation(args.input_radar, args.output_radar)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()