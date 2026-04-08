#!/usr/bin/env python3
"""
Template for data analysis workflows with cfgrib.

Usage:
    python analysis_template.py <input.grib> [--config <config.yaml>]
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GRIBAnalyzer:
    """Template class for GRIB data analysis."""
    
    def __init__(self, filename, config=None):
        """Initialize analyzer.
        
        Args:
            filename: Path to GRIB file
            config: Optional configuration dictionary
        """
        self.filename = filename
        self.config = config or {}
        self.ds = None
    
    def load_data(self):
        """Load GRIB data."""
        print(f"Loading data from {self.filename}...")
        
        # Load with chunking if specified
        chunks = self.config.get('chunks', 'auto')
        self.ds = xr.open_dataset(self.filename, engine='cfgrib', chunks=chunks)
        
        print(f"Loaded dataset with dimensions: {dict(self.ds.dims)}")
        print(f"Variables: {list(self.ds.data_vars.keys())}")
    
    def analyze_variable(self, variable):
        """Analyze a specific variable.
        
        Args:
            variable: Variable name to analyze
        """
        if variable not in self.ds.data_vars:
            raise ValueError(f"Variable '{variable}' not found")
        
        print(f"\nAnalyzing variable: {variable}")
        data = self.ds[variable]
        
        # Basic statistics
        print(f"  Shape: {data.shape}")
        print(f"  Mean: {float(data.mean()):.6f}")
        print(f"  Std: {float(data.std()):.6f}")
        print(f"  Min: {float(data.min()):.6f}")
        print(f"  Max: {float(data.max()):.6f}")
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing > 0:
            print(f"  Missing values: {int(missing)}")
        
        return data
    
    def extract_timeseries(self, variable, lat, lon):
        """Extract time series for a location.
        
        Args:
            variable: Variable name
            lat: Latitude
            lon: Longitude
        
        Returns:
            Time series as pandas Series
        """
        if variable not in self.ds.data_vars:
            raise ValueError(f"Variable '{variable}' not found")
        
        print(f"\nExtracting time series for ({lat}, {lon})...")
        
        ts = self.ds[variable].sel(latitude=lat, longitude=lon, method='nearest')
        ts_series = ts.to_series()
        
        print(f"  Time series length: {len(ts_series)}")
        print(f"  Time range: {ts_series.index[0]} to {ts_series.index[-1]}")
        
        return ts_series
    
    def calculate_spatial_statistics(self, variable):
        """Calculate spatial statistics.
        
        Args:
            variable: Variable name
        
        Returns:
            Dictionary of spatial statistics
        """
        if variable not in self.ds.data_vars:
            raise ValueError(f"Variable '{variable}' not found")
        
        print(f"\nCalculating spatial statistics for {variable}...")
        
        data = self.ds[variable]
        
        stats = {
            'global_mean': float(data.mean()),
            'global_std': float(data.std()),
            'global_min': float(data.min()),
            'global_max': float(data)max()),
        }
        
        # Zonal mean if longitude dimension exists
        if 'longitude' in data.dims:
            zonal_mean = data.mean(dim='longitude')
            stats['zonal_mean'] = float(zonal_mean.mean())
        
        # Meridional mean if latitude dimension exists
        if 'latitude' in data.dims:
            meridional_mean = data.mean(dim='latitude')
            stats['meridional_mean'] = float(meridional_mean.mean())
        
        print(f"  Global mean: {stats['global_mean']:.6f}")
        print(f"  Global std: {stats['global_std']:.6f}")
        
        return stats
    
    def calculate_temporal_statistics(self, variable):
        """Calculate temporal statistics.
        
        Args:
            variable: Variable name
        
        Returns:
            Dictionary of temporal statistics
        """
        if variable not in self.ds.data_vars:
            raise ValueError(f"Variable '{variable}' not found")
        
        if 'time' not in self.ds[variable].dims:
            print(f"Warning: Variable '{variable}' has no time dimension")
            return {}
        
        print(f"\nCalculating temporal statistics for {variable}...")
        
        data = self.ds[variable]
        
        stats = {
            'time_mean': float(data.mean(dim='time').mean()),
            'time_std': float(data.std(dim='time').mean()),
        }
        
        # Daily means if enough data
        if len(data.time) > 24:
            daily = data.resample(time='1D').mean()
            stats['daily_mean'] = float(daily.mean())
        
        print(f"  Time mean: {stats['time_mean']:.6f}")
        print(f"  Time std: {stats['time_std']:.6f}")
        
        return stats
    
    def plot_timeseries(self, variable, lat, lon, output_file=None):
        """Plot time series for a location.
        
        Args:
            variable: Variable name
            lat: Latitude
            lon: Longitude
            output_file: Optional output file path
        """
        ts = self.extract_timeseries(variable, lat, lon)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ts.plot(ax=ax, marker='o', linestyle='-', markersize=4, alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{variable} ({self.ds[variable].attrs.get("units", "")})')
        ax.set_title(f'{variable} at ({lat}°N, {lon}°E)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Plot saved to: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_spatial_field(self, variable, time_idx=0, output_file=None):
        """Plot spatial field.
        
        Args:
            variable: Variable name
            time_idx: Time index
            output_file: Optional output file path
        """
        if variable not in self.ds.data_vars:
            raise ValueError(f"Variable '{variable}' not found")
        
        data = self.ds[variable]
        
        if 'time' in data.dims:
            data = data.isel(time=time_idx)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        var_name = self.ds[variable].attrs.get('long_name', variable)
        var_units = self.ds[variable].attrs.get('units', '')
        
        data.plot(ax=ax, cmap='coolwarm', robust=True,
                 cbar_kwargs={'label': f'{var_name} ({var_units})'})
        
        ax.set_title(f'{var_name} - Time index {time_idx}')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Plot saved to: {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def export_to_netcdf(self, output_file, compression=True):
        """Export dataset to NetCDF.
        
        Args:
            output_file: Output file path
            compression: Whether to use compression
        """
        print(f"\nExporting to NetCDF: {output_file}")
        
        if compression:
            encoding = {var: {'zlib': True, 'complevel': 5} 
                        for var in self.ds.data_vars}
            self.ds.to_netcdf(output_file, encoding=encoding)
        else:
            self.ds.to_netcdf(output_file)
        
        print(f"  Exported successfully")
    
    def close(self):
        """Close dataset."""
        if self.ds is not None:
            self.ds.close()
            print("\nDataset closed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze GRIB data')
    parser.add_argument('input_file', help='Input GRIB file')
    parser.add_argument('--variable', help='Variable to analyze')
    parser.add_argument('--lat', type=float, help='Latitude for time series')
    parser.add_argument('--lon', type=float, help='Longitude for time series')
    parser.add_argument('--config', help='Configuration file')
    parser.add_argument('--output', help='Output file for plots/data')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = GRIBAnalyzer(args.input_file)
    
    try:
        # Load data
        analyzer.load_data()
        
        # Analyze variable if specified
        if args.variable:
            analyzer.analyze_variable(args.variable)
            
            # Extract and plot time series if location specified
            if args.lat and args.lon:
                analyzer.plot_timeseries(args.variable, args.lat, args.lon, args.output)
            
            # Plot spatial field if output specified and no location
            if args.output and not (args.lat and args.lon):
                analyzer.plot_spatial_field(args.variable, output)file=args.output)
        
        # Export to NetCDF if output specified
        if args.output and not args.variable:
            analyzer.export_to_netcdf(args.output)
    
    finally:
        # Close dataset
        analyzer.close()


if __name__ == "__main__":
    main()