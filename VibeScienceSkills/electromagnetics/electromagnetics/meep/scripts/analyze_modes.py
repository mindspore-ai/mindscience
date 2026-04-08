#!/usr/bin/env python3
"""
Mode analysis tools for Meep simulations.

This script provides functions for analyzing resonant modes,
eigenmodes, and mode decomposition in Meep simulations.
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt


def find_resonant_modes(sim, component=mp.Ez, frequency=0.15, 
                         decay_by=0.001, num_modes=10):
    """
    Find resonant modes using Harminv.
    
    Args:
        sim: Meep simulation object
        component: Field component to monitor
        frequency: Center frequency for search
        decay_by: Decay threshold for mode detection
        num_modes: Maximum number of modes to find
    
    Returns:
        Dictionary with mode information
    """
    # Create Harminv object
    harminv = mp.Harminv(component=component,
                       frequency=frequency,
                       decay_by=decay_by)
    
    # Run simulation
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, component, mp.Vector3(0, 0, 0), 1e-3))
    
    # Get modes
    modes = harminv.modes
    
    mode_data = []
    for i, mode in enumerate(modes):
        q_factor = compute_q_factor(mode.freq, mode.decay)
        mode_data.append({
            'index': i,
            'frequency': mode.freq,
            'decay_rate': mode.decay,
            'q_factor': q_factor
        })
    
    return {
        'modes': mode_data,
        'num_modes': len(mode_data),
        'harminv': harminv
    }


def compute_q_factor(frequency, decay_rate):
    """
    Compute Q factor from frequency and decay rate.
    
    Args:
        frequency: Resonant frequency
        decay_rate: Decay rate
    
    Returns:
        Q factor
    """
    if decay_rate == 0:
        return np.inf
    return frequency / (2 * decay_rate)


def get_eigenmode(sim, k, component=mp.Ez, center=None, size=None):
    """
    Get eigenmode for waveguide or periodic structure.
    
    Args:
        sim: Meep simulation object
        k: Wave vector (Vector3)
        component: Field component
        center: Center of mode region
        size: Size of mode region
    
    Returns:
        Eigenmode object
    """
    if center is None:
        center = mp.Vector3()
    if size is None:
        size = sim.cell_size
    
    mode = sim.get_eigenmode(k(k), component, center, size)
    return mode


def analyze_waveguide_modes(sim, fcen, df, nfreq, k_range, 
                          component=mp.Ez, center=None, size=None):
    """
    Analyze waveguide modes over a range of propagation constants.
    
    Args:
        sim: Meep simulation object
        fcen: Center frequency
        df: Frequency width
        nfreq: Number of frequency points
        k_range: Range of propagation constants (k_min, k_max, num_k)
        component: Field component
        center: Center of mode region
        size: Size of mode region
    
    Returns:
        Dictionary with mode analysis
    """
    if center is None:
        center = mp.Vector3()
    if size is None:
        size = sim.cell_size
    
    k_min, k_max, num_k = k_range
    k_values = np.linspace(k_min, k_max, num_k)
    
    modes = []
    for i, k in enumerate(k_values):
        try:
            mode = sim.get_eigenmode(mp.Vector3(k, 0, 0), component, center, size)
            modes.append({
                'index': i,
                'k': k,
                'frequency': mode.freq,
                'group_velocity': mode.freq / k
            })
        except:
            modes.append({
                'index': i,
                'k': k,
                'frequency': np.nan,
                'group_velocity': np.nan
            })
    
    return {
        'modes': modes,
        'k_values': k_values
    }


def decompose_fields(sim, k, mode, flux_region, fcen, df, nfreq):
    """
    Decompose fields into waveguide modes.
    
    Args:
        sim: Meep simulation object
        k: Wave vector
        mode: Eigenmode object
        flux_region: FluxRegion for decomposition
        fcen: Center frequency
        df: Frequency width
        nfreq: Number of frequency points
    
    Returns:
        Dictionary with mode decomposition results
    """
    # Add mode monitor
    d = sim.add_mode_monitor(k, mode, flux_region)
    
    # Run simulation
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(0, 0, 0), 1e-3))
    
    # Get mode coefficients
    mode_coeffs = sim.get_mode_coeffs(d)
    freqs = mp.get_flux_freqs(d)
    
    return {
        'mode_coefficients': mode_coeffs,
        'frequencies': freqs
    }


def plot_mode_profile(mode, center=None, size=None, cmap='RdBu'):
    """
    Plot spatial profile of a resonant mode.
    
    Args:
        mode: Mode object (from Harminv or Eigenmode)
        center: Center of plotting region
        size: Size of plotting region
        cmap: Colormap for mode field
    
    Returns:
        matplotlib figure object
    """
    # Get mode field
    mode_field = mode.get_field()
    
    if center is None:
        center = mp.Vector3()
    if size is None:
        size = mode_field.shape
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mode_field.transpose(), cmap=cmap, interpolation='spline36')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mode Amplitude')
    
    return fig


def plot_mode_spectrum(modes, xlabel='Frequency', 
                        ylabel='Q Factor', title='Mode Spectrum'):
    """
    Plot resonant mode spectrum.
    
    Args:
        modes: List of mode dictionaries
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
    
    Returns:
        matplotlib figure object
    """
    frequencies = [mode['frequency'] for mode in modes]
    q_factors = [mode['q_factor'] for mode in modes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frequencies, q_factors, 'bo-', markersize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_dispersion_curve(modes, xlabel='Propagation Constant k', 
                           ylabel='Frequency', title='Dispersion Curve'):
    """
    Plot dispersion curve for waveguide modes.
    
    Args:
        modes: List of mode dictionaries
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
    
    Returns:
        matplotlib figure object
    """
    k_values = [mode['k'] for mode in modes]
    frequencies = [mode['frequency'] for mode in modes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, frequencies, 'bo-', markersize=8, label='Dispersion')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig


def compute_modal_volume(sim, mode_field, center=None, size=None):
    """
    Compute modal volume of a resonant mode.
    
    Args:
        sim: Meep simulation object
        mode_field: Mode field array
        center: Center of integration region
        size: Size of integration region
    
    Returns:
        Modal volume
    """
    if center is None:
        center = mp.Vector3()
    if size is None:
        size = sim.cell_size
    
    # Get dielectric function
    eps = sim.get_array(center=center, size=size, component=mp.Dielectric)
    
    # Compute modal volume
    numerator = np.sum(np.abs(mode_field)**2 * eps)
    denominator = np.sum(np.abs(mode_field)**2)
    
    if denominator == 0:
        return np.inf
    
    modal_volume = numerator / denominator
    return modal_volume


def compute_purcell_factor(sim, mode_field, center=None, size=None):
    """
    Compute Purcell factor for a resonant mode.
    
    Args:
        sim: Meep simulation object
        mode_field: Mode field array
        center: Center of integration region
        size: Size of integration region
    
    Returns:
        Purcell factor
    """
    modal_volume = compute_modal_volume(sim, mode_field, center, size)
    frequency = 0.15  # Approximate frequency
    
    # Compute wavelength in vacuum
    wavelength = 1.0 / frequency
    
    # Compute Purcell factor
    purcell_factor = modal_volume / (wavelength**3)
    
    return purcell_factor


def compare_modes(mode1, mode2):
    """
    Compare two resonant modes.
    
    Args:
        mode1: First mode dictionary
        mode2: Second mode dictionary
    
    Returns:
        Dictionary with comparison results
    """
    freq_diff = abs(mode1['frequency'] - mode2['frequency'])
    q_diff = abs(mode1['q_factor'] - mode2['q_factor'])
    
    return {
        'frequency_difference': freq_diff,
        'q_factor_difference': q_diff,
        'frequency_ratio': mode1['frequency'] / mode2['frequency'],
        'q_factor_ratio': mode1['q_factor'] / mode2['q_factor']
    }


def sort_modes_by_frequency(modes, ascending=True):
    """
    Sort modes by frequency.
    
    Args:
        modes: List of mode dictionaries
        ascending: Sort order
    
    Returns:
        Sorted list of modes
    """
    return sorted(modes, key=lambda x: x['frequency'], reverse=not ascending)


def sort_modes_by_q_factor(modes, ascending=True):
    """
    Sort modes by Q factor.
    
    Args:
        modes: List of mode dictionaries
        ascending: Sort order
    
    Returns:
        Sorted list of modes
    """
    return sorted(modes, key=lambda x: x['q_factor'], reverse=not ascending)


def main():
    """Example usage of mode analysis functions."""
    print("Meep Mode Analysis Tools")
    print("=========================")
    print()
    print("Available functions:")
    print("- find_resonant_modes: Find resonant modes using Harminv")
    print("- compute_q_factor: Compute Q factor")
    print("- get_eigenmode: Get eigenmode for waveguide")
    print("- analyze_waveguide_modes: Analyze waveguide modes")
    print("- decompose_fields: Decompose fields into modes")
    print("- plot_mode_profile: Plot mode spatial profile")
    print("- plot_mode_spectrum: Plot mode spectrum")
    print("- plot_dispersion_curve: Plot dispersion curve")
    print("- compute_modal_volume: Compute modal volume")
    print("- compute_purcell_factor: Compute Purcell factor")
    print("- compare_modes: Compare two modes")
    print("- sort_modes_by_frequency: Sort modes by frequency")
    print("- sort_modes_by_q_factor: Sort modes by Q factor")
    print()
    print("Example usage:")
    print("  import meep as mp")
    print("  from analyze_modes import find_resonant_modes, plot_mode_spectrum")
    print("  ")
    print("  sim = mp.Simulation(...)")
    print("  ")
    print("  # Find resonant modes")
    print("  results = find_resonant_modes(sim, frequency=0.15)")
    print("  ")
    print("  # Plot mode spectrum")
    print("  fig = plot_mode_spectrum(results['modes'])")
    print("  plt.show()")


if __name__ == "__main__":
    main()