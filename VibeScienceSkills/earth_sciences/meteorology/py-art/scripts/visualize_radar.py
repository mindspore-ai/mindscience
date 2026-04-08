#!/usr/bin/env python3
"""
Create visualizations of radar data.

Usage:
    python visualize_radar.py <radar_file> --field <field> --output <output.png>
    
Examples:
    python visualize_radar.py radar.nc --field reflectivity --output ppi.png
    python visualize_radar.py radar.nc --field velocity --output rhi.png
"""

import sys
import argparse
import pyart
import matplotlib.pyplot as plt


def plot_ppi(radar_file, field_name, output_file):
    """Create PPI (Plan Position Indicator) plot."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(radar_file)
        
        # Check if field exists
        if field_name not in radar.fields:
            print(f"Error: Field '{field_name}' not found")
            print(f"Available fields: {list(radar.fields.keys())}")
            return False
        
        # Create display
        display = pyart.graph.RadarDisplay(radar)
        
        # Set color limits based on field
        if field_name == 'reflectivity':
            vmin, vmax = -30, 70
            cmap = 'NWSpectral'
        elif field_name == 'velocity':
            vmin, vmax = -30, 30
            cmap = 'RdBu_r'
        elif field_name == 'spectrum_width':
            vmin, vmax = 0, 10
            cmap = 'viridis'
        else:
            vmin, vmax = None, None
            cmap = 'jet'
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
        # Plot PPI
        display.plot(field_name, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)
        display.set_limits(xlim=(-50, 50), ylim=(-50, 50))
        
        # Add colorbar
        plt.colorbar(ax.collections[0], ax=ax, label=field_name)
        
        # Add title
        ax.set_title(f'{field_name} PPI')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PPI plot saved to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def plot_rhi(radar_file, field_name, sweep_num, output_file):
    """Create RHI (Range-Height Indicator) plot."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(radar_file)
        
        # Check if field exists
        if field_name not in radar.fields:
            print(f"Error: Field '{field_name}' not found")
            return False
        
        # Create display
        display = pyart.graph.RadarDisplay(radar)
        
        # Set color limits based on field
        if field_name == 'reflectivity':
            vmin, vmax = -30, 70
            cmap = 'NWSpectral'
        elif field_name == 'velocity':
            vmin, vmax = -30, 30
            cmap = 'RdBu_r'
        else:
            vmin, vmax = None, None
            cmap = 'jet'
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Plot RHI
        display.plot_rhi(field_name, sweep_num, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)
        
        # Add colorbar
        plt.colorbar(ax.collections[0], ax=ax, label=field_name)
        
        # Add title
        ax.set_title(f'{field_name} RHI - Sweep {sweep_num}')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"RHI plot saved to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def plot_cappi(radar_file, field_name, altitude, output_file):
    """Create CAPPI (Constant Altitude PPI) plot."""
    
    try:
        # Read radar data
        radar = pyart.io.read_arm_netcdf(radar_file)
        
        # Check if field exists
        if field_name not in radar.fields:
            print(f"Error: Field '{field_name}' not found")
            return False
        
        # Create display
        display = pyart.graph.RadarDisplay(radar)
        
        # Set color limits based on field
        if field_name == 'reflectivity':
            vmin, vmax = -30, 70
            cmap = 'NWSpectral'
        else:
            vmin, vmax = None, None
            cmap = 'jet'
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
        # Plot CAPPI
        display.plot_ppi(field_name, altitude, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)
        display.set_limits(xlim=(-50, 50), ylim=(-50, 50))
        
        # Add colorbar
        plt.colorbar(ax.collections[0], ax=ax, label=field_name)
        
        # Add title
        ax.set_title(f'{field_name} CAPPI - {altitude} m')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"CAPPI plot saved to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Visualize radar data')
    parser.add_argument('radar_file', help='Input radar file')
    parser.add_argument('--field', required=True, help='Field name to plot')
    parser.add_argument('--output', required=True, help='Output PNG file')
    parser.add_argument('--sweep', type=int, help='Sweep number for RHI plot')
    parser.add_argument('--altitude', type=float, help='Altitude for CAPPI plot (m)')
    parser.add_argument('--plot-type', choices=['ppi', 'rhi', 'cappi'], 
                       default='ppi', help='Plot type')
    
    args = parser.parse_args()
    
    success = False
    
    if args.plot_type == 'ppi':
        success = plot_ppi(args.radar_file, args.field, args.output)
    elif args.plot_type == 'rhi':
        if args.sweep is None:
            print("Error: Must specify --sweep for RHI plot")
            sys.exit(1)
        success = plot_rhi(args.radar_file, args.field, args.sweep, args.output)
    elif args.plot_type == 'cappi':
        if args.altitude is None:
            print("Error: Must specify --altitude for CAPPI plot")
            sys.exit(1)
        success = plot_cappi(args.radar_file, args.field, args.altitude, args.output)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()