#!/usr/bin/env python3
"""
Analyze ensemble forecast data.

Usage:
    python ensemble_analysis.py <ensemble.grib> [--statistics] [--spread] [--probabilities]
    
Examples:
    python ensemble_analysis.py ensemble.grib
    python ensemble_analysis.py ensemble.grib --statistics --spread
"""

import sys
import argparse
import xarray as xr
import numpy as np


def calculate_ensemble_statistics(ds, variable='t2m'):
    """Calculate ensemble statistics."""
    
    if variable not in ds.data_vars:
        print(f"Error: Variable '{variable}' not found")
        return False
    
    if 'number' not in ds.dims:
        print("Error: Dataset does not have ensemble dimension 'number'")
        return False
    
    var = ds[variable]
    
    print(f"\n=== Ensemble Statistics for {variable} ===")
    
    # Ensemble mean
    ensemble_mean = var.mean(dim='number')
    print(f"Ensemble mean: {float(ensemble_mean.mean()):.6f}")
    
    # Ensemble spread (standard deviation)
    ensemble_spread = var.std(dim='number')
    print(f"Ensemble spread: {float(ensemble_spread.mean()):.6f}")
    
    # Ensemble minimum and maximum
    ensemble_min = var.min(dim='number')
    ensemble_max = var.max(dim='number')
    print(f"Ensemble min: {float(ensemble_min.min()):.6f}")
    print(f"Ensemble max: {float(ensemble_max.max()):.6f}")
    
    # Ensemble range
    ensemble_range = ensemble_max - ensemble_min
    print(f"Ensemble range: {float(ensemble_range.mean()):.6f}")
    
    return True


def calculate_ensemble_spread(ds, variable='t2m'):
    """Calculate detailed ensemble spread."""
    
    if variable not in ds.data_vars:
        print(f"Error: Variable '{variable}' not found")
        return False
    
    if 'number' not in ds.dims:
        print("Error: Dataset does not have ensemble dimension 'number'")
        return False
    
    var = ds[variable]
    
    print(f"\n=== Ensemble Spread for {variable} ===")
    
    # Calculate ensemble spread
    spread = var.std(dim='number')
    
    # Spatial distribution of spread
    print(f"Mean spread: {float(spread.mean()):.6f}")
    print(f"Max spread: {float(spread.max()):.6f}")
    print(f"Min spread: {float(spread.min()):.6f}")
    
    # Temporal distribution of spread
    if 'time' in spread.dims:
        temporal_spread = spread.mean(dim=['latitude', 'longitude'])
        print(f"\nTemporal spread:")
        print(f"  Mean: {float(temporal_spread.mean()):.6f}")
        print(f"  Max: {float(temporal_spread.max()):.6f}")
        print(f"  Min: {float(temporal_spread.min()):.6f}")
    
    return True


def calculate_probabilities(ds, variable='t2m', threshold=300):
    """Calculate probabilities of exceeding thresholds."""
    
    if variable not in ds.data_vars:
        print(f"Error: Variable '{variable}' not found")
        return False
    
    if 'number' not in ds.dims:
        print("Error: Dataset does not have ensemble dimension 'number'")
        return False
    
    var = ds[variable]
    
    print(f"\n=== Probabilities for {variable} ===")
    
    # Probability of exceeding threshold
    prob_above = (var > threshold).mean(dim='number')
    print(f"Probability > {threshold}: {float(prob_above.mean()):.4f}")
    
    # Probability of being below threshold
    prob_below = (var < threshold).mean(dim='number')
    print(f"Probability < {threshold}: {float(prob_below.mean()):.4f}")
    
    # Spatial distribution
    if 'latitude' in prob_above.dims and 'longitude' in prob_above.dims:
        print(f"\nSpatial probability distribution:")
        print(f"  Max probability > {threshold}: {float(prob_above.max()):.4f}")
        print(f"  Min probability > {threshold}: {float(prob_above.min()):.4f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Analyze ensemble forecast data')
    parser.add_argument('input_file', help='Input GRIB file with ensemble data')
    parser.add_argument('--variable', default='t2m', help='Variable name (default: t2m)')
    parser.add_argument('--statistics', action='store_true', help='Calculate ensemble statistics')
    parser.add_argument('--spread', action='store_true', help='Calculate ensemble spread')
    parser.add_argument('--probabilities', action='store_true', help='Calculate probabilities')
    parser.add_argument('--threshold', type=float, default=300, help='Threshold for probability calculations')
    
    args = parser.parse_args()
    
    if not (args.statistics or args.spread or args.probabilities):
        args.statistics = True
    
    try:
        with xr.open_dataset(args.input_file, engine='cfgrib') as ds:
            if args.statistics:
                calculate_ensemble_statistics(ds, args.variable)
            
            if args.spread:
                calculate_ensemble_spread(ds, args.variable)
            
            if args.probabilities:
                calculate_probabilities(ds, args.variable, args.threshold)
    
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()