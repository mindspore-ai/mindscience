#!/usr/bin/env python3
"""
Convert radar data to Cartesian coordinates.

Usage:
    python convert_coordinates.py <input_radar> <output_grid> --range1 <km> --range2 <km> --grid-size <size>
    
Examples:
    python convert_coordinates.py radar.nc grid.nc --range1 0 --range2 50 --grid-size 101
"""

import sys
import argparse
import pyart


def convert_to_cartesian(input_file, output_file, range1_km, range2_km, grid_size):
    """Convert polar radar data to Cartesian grid."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Convert to Cartesian grid
        grid = pyart.map.grid_constant_azimuth_range(
            radar, 
            range_1_km=range1_km, 
            range_2_km=range2_km, 
            grid_shape=(grid_size, grid_size),
            fields=['reflectivity', 'velocity'],
            edge_factor=0.0
        )
        
        # Write output
        pyart.io.write_arm_netcdf(grid, output_file)
        
        print(f"Conversion completed successfully")
        print(f"Input radar: {radar.nsweeps} sweeps, {radar.ngates} gates")
        print(f"Output grid: {grid_size} x {grid_size}")
        print(f"Output file: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def convert_sector(input_file, output_file, center_angle, width_angle, range1_km, range2_km, grid_size):
    """Convert sector to Cartesian grid."""
    
    try:
        # radar data
        radar = pyart.io.read_arm_netcdf(input_file)
        
        # Convert sector
        grid = pyart.map.grid_from_sector(
            radar, 
            center_angle=center_angle, 
            width_angle=width_angle,
            range_1_km=range1_km,
            range_2_km=range2_km,
            grid_shape=(grid_size, grid_size),
            fields=['reflectivity']
        )
        
        # Write output
        pyart.io.write_arm_netcdf(grid, output_file)
        
        print(f"Sector conversion completed successfully")
        print(f"Output file: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert radar data to Cartesian coordinates')
    parser.add_argument('input_radar', help='Input radar file')
    parser.add_argument('output_grid', help='Output grid file')
    parser.add_argument('--range1', type=float, required=True, help='Start range (km)')
    parser.add_argument('--range2', type=float, required=True, help='End range (km)')
    parser.add_argument('--grid-size', type=int, required=True, help='Grid size')
    parser.add_argument('--center-angle', type=float, help='Center angle for sector (degrees)')
    parser.add_argument('--width-angle', type=float, help='Width angle for sector (degrees)')
    
    args = parser.parse_args()
    
    success = False
    
    if args.center_angle and args.width_angle:
        success = convert_sector(args.input_radar, args.output_grid, 
                             args.center_angle, args.width_angle,
                             args.range1, args.range2, args.grid_size)
    else:
        success = convert_to_cartesian(args.input_radar, args.output_grid,
                                     args.range1, args.range2, args.grid_size)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()