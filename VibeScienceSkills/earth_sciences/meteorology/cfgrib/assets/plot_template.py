#!/usr/bin/env python3
"""
Template for creating custom plots with cfgrib data.

Usage:
    python plot_template.py <input.grib> --variable <var> --output <output.png>
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def create_custom_plot(input_file, variable, output_file):
    """Create a custom plot of GRIB data."""
    
    with xr.open_dataset(input_file, engine='cfgrib') as ds:
        # Check if variable exists
        if variable not in ds.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")
        
        # Select data
        data = ds[variable]
        
        # Select time if available
        if 'time' in data.dims:
            data = data.isel(time=0)
        
        # Create figure with map projection
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Add map features
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
        
        # Get variable name and units
        var_name = data.attrs.get('long_name', variable)
        var_units = data.attrs.get('units', '')
        
        # Plot data
        data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='coolwarm',
            robust=True,
            cbar_kwargs={
                'label': f'{var_name} ({var_units})',
                'shrink': 0.8,
                'orientation': 'horizontal'
            }
        )
        
        # Set title
        ax.set_title(f'{var_name}', fontsize=14, fontweight='bold')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create custom plot of GRIB data')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('--variable', required=True, help='Variable name')
    parser.add_argument('--output', required=True, help='Output PNG file')
    
    args = parser.parse_args()
    
    create_custom_plot(args.input_file, args.variable, args.output)