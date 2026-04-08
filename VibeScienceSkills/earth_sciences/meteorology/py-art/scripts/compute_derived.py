#!/usr/bin/env python3
"""
Compute derived quantities from radar data.

Usage:
    python compute_derived.py <input_radar> <output_radar> --zh --zdr --kdp
    
Examples:
    python compute_derived.py radar.nc radar_derived.nc --zh --zdr --kdp
"""

import sys
import argparse
import pyart


def compute_zh(input_file, output_file):
    """Compute horizontal reflectivity (ZH)."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Compute ZH
        radar_zh = pyart.retrieve.compute_zh(radar)
        
        # Write output
        pyart.io.write_arm_netcdf(radar_zh, output_file)
        
        print("ZH computed successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def compute_zdr(input_file, output_file):
    """Compute differential reflectivity (ZDR)."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Compute ZDR
        radar_zdr = pyart.retrieve.compute_zdr(radar)
        
        # Write output
        pyart.io.write_arm_netcdf(radar_zdr, output_file)
        
        print("ZDR computed successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def compute_kdp(input_file, output_file):
    """Compute specific differential phase (KDP)."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Compute KDP
        radar_kdp = pyart.retrieve.compute_kdp(radar)
        
        # Write output
        pyart.io.write_arm_netcdf(radar_kdp, output_file)
        
        print("KDP computed successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception Exception as e:
        print(f"Error: {e}")
        return False


def compute_rain_rate_zr(input_file, output_file):
    """Estimate rain rate using Z-R relationship."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Estimate rain rate
        radar_rain = pyart.retrieve.est_rain_rate_zr(radar)
        
        # Write output
        pyart.io.write_arm_netcdf(radar_rain, output_file)
        
        print("Rain rate (Z-R) estimated successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def compute_rain_rate_kdp(input_file, output_file):
    """Estimate rain rate using KDP."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Estimate rain rate from KDP
        radar_rain = pyart.retrieve.est_rain_rate_kdp(radar)
        
        # Write output
        pyart.io.write_arm_netcdf(radar_rain, output_file)
        
        print("Rain rate (KDP) estimated successfully")
        print(f"Output written to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Compute derived quantities from radar data')
    parser.add_argument('input_radar', help='Input radar file')
    parser.add_argument('output_radar', help='Output radar file')
    parser.add_argument('--zh', action='store_true', help='Compute ZH')
    parser.add_argument('--zdr', action='store_true', help='Compute ZDR')
    parser.add_argument('--kdp', action='store_true', help='Compute KDP')
    parser.add_argument('--rain-zr', action='store_true', help='Estimate rain rate (Z-R)')
    parser.add_argument('--rain-kdp', action='store_true', help='Estimate rain rate (KDP)')
    
    args = parser.parse_args()
    
    if not (args.zh or args.zdr or args.kdp or args.rain_zr or args.rain_kdp):
        print("Error: Must specify at least one derived quantity")
        sys.exit(1)
    
    success = False
    
    if args.zh:
        success = compute_zh(args.input_radar, args.output_radar)
    
    if args.zdr:
        success = compute_zdr(args.input_radar, args.output_radar)
    
    if args.kdp:
        success = compute_kdp(args.input_radar, args.output_radar)
    
    if args.rain_zr:
        success = compute_rain_rate_zr(args.input_radar, args.output_radar)
    
    if args.rain_kdp:
        success = compute_rain_rate_kdp(args.input_radar, args.output_radar)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()