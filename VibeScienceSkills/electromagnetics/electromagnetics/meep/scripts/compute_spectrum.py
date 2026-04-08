#!/usr/bin/env python3
"""
Spectrum computation helpers for Meep simulations.

This script provides functions for computing transmission, reflection,
and scattering spectra from Meep simulations.
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt


def compute_spectrum(sim, flux_regions, fcen, df, nfreq, 
                     incident_flux=None, refl_data=None):
    """
    Compute transmission and reflection spectra.
    
    Args:
        sim: Meep simulation object
        flux_regions: List of FluxRegion objects
        fcen: Center frequency
        df: Frequency width
        nfreq: Number of frequency points
        incident_flux: Incident flux spectrum (for normalization)
        refl_data: Reflection flux data (for subtraction)
    
    Returns:
        Dictionary with spectra and frequencies
    """
    # Add flux monitors
    fluxes = []
    for region in flux_regions:
        flux = sim.add_flux(fcen, df, nfreq, region)
        fluxes.append(flux)
    
    # Load reflection data if provided
    if refl_data is not None and len(fluxes) > 0:
        sim.load_minus_flux_data(fluxes[0], refl_data)
    
    # Run simulation
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(5, 0), 1e-3))
    
    # Get flux spectra
    flux_spectra = [mp.get_fluxes(flux) for flux in fluxes]
    freqs = mp.get_flux_freqs(fluxes[0])
    
    # Normalize if incident flux provided
    if incident_flux is not None:
        normalized_spectra = []
        for flux_spectrum in flux_spectra:
            normalized = flux_spectrum / incident_flux
            normalized_spectra.append(normalized)
        flux_spectra = normalized_spectra
    
    return {
        'flux_spectra': flux_spectra,
        'frequencies': freqs
    }


def compute_two_run_spectrum(sim_ref, sim_actual, flux_regions, fcen, df, nfreq):
    """
    Compute spectrum using two-run method for accurate normalization.
    
    Args:
        sim_ref: Reference simulation (without scatterer)
        sim_actual: Actual simulation (with scatterer)
        flux_regions: List of FluxRegion objects
        fcen: Center frequency
        df: Frequency width
        nfreq: Number of frequency points
    
    Returns:
        Dictionary with transmission, reflection, and loss spectra
    """
    # First run: reference simulation
    print("Running reference simulation...")
    fluxes_ref = []
    for region in flux_regions:
        flux = sim_ref.add_flux(fcen, df, nfreq, region)
        fluxes_ref.append(flux)
    
    sim_ref.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(5, 0), 1e-3))
    
    # Save incident flux and reflection data
    incident_flux = mp.get_fluxes(fluxes_ref[-1])  # Last flux is transmission
    refl_data = sim_ref.get_flux_data(fluxes_ref[0])  # First flux is reflection
    
    # Second run: actual simulation
    print("Running actual simulation...")
    fluxes_actual = []
    for region in flux_regions:
        flux = sim_actual.add_flux(fcen, df, nfreq, region)
        fluxes_actual.append(flux)
    
    # Load negative reflection data
    sim_actual.load_minus_flux_data(fluxes_actual[0], refl_data)
    
    sim_actual.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(5, 0), 1e-3))
    
    # Get fluxes
    refl_flux = mp.get_fluxes(fluxes_actual[0])
    tran_flux = mp.get_fluxes(fluxes_actual[-1])
    freqs = mp.get_flux_freqs(fluxes_actual[0])
    
    # Compute spectra
    transmittance = tran_flux / incident_flux
    reflectance = -refl_flux / incident_flux
    loss = 1 - transmittance - reflectance
    
    return {
        'transmittance': transmittance,
        'reflectance': reflectance,
        'loss': loss,
        'frequencies': freqs,
        'incident_flux': incident_flux
    }


def compute_scattering_spectrum(sim, r, fcen, df, nfreq):
    """
    Compute scattering cross section spectrum.
    
    Args:
        sim: Meep simulation object
        r: Radius of scatterer
        fcen: Center frequency
        df: Frequency width
        nfreq: Number of frequency points
    
    Returns:
        Dictionary with scattering cross section
    """
    # Create flux box around scatterer
    box_x1 = sim.add_flux(fcen, df, nfreq, 
                         mp.FluxRegion(center=mp.Vector3(x=-r), 
                                       size=mp.Vector3(0, 2*r, 2*r)))
    box_x2 = sim.add_flux(fcen, df, nfreq, 
                         mp.FluxRegion(center=mp.Vector3(x=+r), 
                                       size=mp.Vector3(0, 2*r, 2*r)))
    box_y1 = sim.add_flux(fcen, df, nfreq, 
                         mp.FluxRegion(center=mp.Vector3(y=-r), 
                                       size=mp.Vector3(2*r, 0, 2*r)))
    box_y2 = sim.add_flux(fcen, df, nfreq, 
                         mp.FluxRegion(center=mp.Vector3(y=+r), 
                                       size=mp.Vector3(2*r, 0, 2*r)))
    box_z1 = sim.add_flux(fcen, df, nfreq, 
                         mp.FluxRegion(center=mp.Vector3(z=-r), 
                                       size=mp.Vector3(2*r, 2*r, 0)))
    box_z2 = sim.add_flux(fcen, df, nfreq, 
                         mp.FluxRegion(center=mp.Vector3(z=+r), 
                                       size=mp.Vector3(2*r, 2*r, 0)))
    
    # Run simulation
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(0, 0), 1e-3))
    
    # Get scattered power
    scattered_power = (sum(mp.get_fluxes(box_x1)) + 
                     sum(mp.get_fluxes(box_x2)) + 
                     sum(mp.get_fluxes(box_y1)) + 
                     sum(mp.get_fluxes(box_y2)) + 
                     sum(mp.get_fluxes(box_z1)) + 
                     sum(mp.get_fluxes(box_z2)))
    
    freqs = mp.get_flux_freqs(box_x1)
    
    return {
        'scattered_power': scattered_power,
        'frequencies': freqs
    }


def plot_spectrum(spectra, xlabel='Wavelength (μm)', 
                  ylabel='Normalized Power', title='Spectrum'):
    """
    Plot spectrum data.
    
    Args:
        spectra: Dictionary with spectrum data
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
    
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each spectrum
    for key, data in spectra.items():
        if key == 'frequencies':
            continue
        ax.plot(spectra['frequencies'], data, label=key)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_trl_spectrum(transmittance, reflectance, loss, freqs, 
                      xlabel='Wavelength (μm)', title='TRL Spectrum'):
    """
    Plot transmission, reflection, and loss spectra.
    
    Args:
        transmittance: Transmittance spectrum
        reflectance: Reflectance spectrum
        loss: Loss spectrum
        freqs: Frequency array
        xlabel: X-axis label
        title: Plot title
    
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(freqs, transmittance, 'r-', linewidth=2, label='Transmittance')
    ax.plot(freqs, reflectance, 'b-', linewidth=2, label='Reflectance')
    ax.plot(freqs, loss, 'g-', linewidth=2, label='Loss')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Normalized Power')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    return fig


def find_peaks(spectrum, freqs, min_prominence=0.1, min_distance=5):
    """
    Find peaks in spectrum.
    
    Args:
        spectrum: Spectrum array
        freqs: Frequency array
        min_prominence: Minimum prominence for peak detection
        min_distance: Minimum distance between peaks
    
    Returns:
        List of (frequency, value) tuples
    """
    from scipy.signal import find_peaks
    
    peaks, properties = find_peaks(spectrum, 
                                       prominence=min_prominence,
                                       distance=min_distance)
    
    peak_data = [(freqs[i], spectrum[i]) for i in peaks]
    
    return peak_data


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


def analyze_modes(harminv):
    """
    Analyze resonant modes from Harminv object.
    
    Args:
        harminv: Harminv object
    
    Returns:
        Dictionary with mode analysis
    """
    modes = harminv.modes
    
    mode_data = []
    for mode in modes:
        q_factor = compute_q_factor(mode.freq, mode.decay)
        mode_data.append({
            'frequency': mode.freq,
            'decay_rate': mode.decay,
            'q_factor': q_factor
        })
    
    return {
        'modes': mode_data,
        'num_modes': len(mode_data)
    }


def main():
    """Example usage of spectrum computation functions."""
    print("Meep Spectrum Computation Helpers")
    print("=====================================")
    print()
    print("Available functions:")
    print("- compute_spectrum: Compute transmission/reflection spectra")
    print("- compute_two_run_spectrum: Two-run method for accurate spectra")
    print("- compute_scattering_spectrum: Compute scattering cross section")
    print("- plot_spectrum: Plot spectrum data")
    print("- plot_trl_spectrum: Plot TRL spectra")
    print("- find_peaks: Find peaks in spectrum")
    print("- compute_q_factor: Compute Q factor")
    print("- analyze_modes: Analyze resonant modes")
    print()
    print("Example usage:")
    print("  import meep as mp")
    print("  from compute_spectrum import compute_two_run_spectrum, plot_trl_spectrum")
    print("  ")
    print("  # Setup simulations")
    print("  sim_ref = mp.Simulation(...)")
    print("  sim_actual = mp.Simulation(...)")
    print("  ")
    print("  # Compute spectra")
    print("  fcen = 0.15")
    print("  df = 0.1")
    print("  nfreq = 100")
    print("  flux_regions = [mp.FluxRegion(...), mp.FluxRegion(...)]")
    print("  ")
    print("  results = compute_two_run_spectrum(sim_ref, sim_actual, ")
    print("                                       flux_regions, fcen, df, nfreq)")
    print("  ")
    print("  # Plot spectra")
    print("  fig = plot_trl_spectrum(results['transmittance'], ")
    print("                          results['reflectance'], ")
    print("                          results['loss'], ")
    print("                          results['frequencies'])")
    print("  plt.show()")


if __name__ == "__main__":
    main()