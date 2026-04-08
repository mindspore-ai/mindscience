#!/usr/bin/env python3
"""
Network analysis tools for scikit-rf.

This script provides functions for network analysis, parameter
extraction, and network properties calculation.
"""

import skrf as rf
import numpy as np
import matplotlib.pyplot as plt


def analyze_network_properties(network):
    """
    Analyze basic network properties.
    
    Args:
        network: Network object
    
    Returns:
        Dictionary with network properties
    """
    properties = {
        'name': network.name,
        'nports': network.nports,
        'nfreq': network.nfreq,
        'frequency_range': (network.frequency.f_scaled[0], 
                            network.frequency.f_scaled[-1]),
        'is_reciprocal': network.is_reciprocal(),
        'is_symmetric': network.is_symmetric(),
        'is_lossless': network.is_lossless()
    }
    
    return properties


def extract_s_parameters(network):
    """
    Extract all S-parameters from network.
    
    Args:
        network: Network object
    
    Returns:
        Dictionary with S-parameters
    """
    s_params = {
        's': network.s,
        's_mag': network.s_mag,
        's_deg': network.s_deg,
        's_rad': network.s_rad,
        's_re': network.s_re,
        's_im': network.s_im,
        's11': network.s11,
        's12': network.s12,
        's21': network.s21,
        's22': network.s22
    }
    
    return s_params


def extract_z_parameters(network):
    """
    Extract Z-parameters from network.
    
    Args:
        network: Network object
    
    Returns:
        Dictionary with Z-parameters
    """
    z_params = {
        'z': network.z,
        'z0': network.z0,
        'z_re': network.z_re,
        'z_im': network.z_im,
        'z_mag': network.z_mag,
        'z_ang': network.z_ang
    }
    
    return z_params


def calculate_return_loss(network):
    """
    Calculate return loss from S-parameters.
    
    Args:
        network: Network object
    
    Returns:
        Return loss array
    """
    # Calculate return loss: RL = 1 - |S11|^2 - |S21|^2
    s11_mag = np.abs(network.s11)**2
    s21_mag = np.abs(network.s21)**2
    return_loss = 1 - s11_mag - s21_mag
    
    return return_loss


def calculate_insertion_loss(network):
    """
    Calculate insertion loss from S-parameters.
    
    Args:
        network: Network object
    
    Returns:
        Insertion loss array
    """
    # Calculate insertion loss: IL = -20*log10(|S21|)
    s21_mag = np.abs(network.s21)
    insertion_loss = -20 * np.log10(s21_mag)
    
    return insertion_loss


def calculate_vswr(network):
    """
    Calculate VSWR from S-parameters.
    
    Args:
        network: Network object
    
    Returns:
        VSWR array
    """
    # Calculate VSWR: VSWR = 20*log10(|S11|^2/(1-|S11|^2))
    s11_mag = np.abs(network.s11)**2
    vswr = 20 * np.log10(s11_mag / (1 - s11_mag))
    
    return vswr


def find_extremum_values(network, parameter='s11', mode='max'):
    """
    Find extremum values of network parameter.
    
    Args:
        network: Network object
        parameter: Parameter name ('s11', 's21', etc.)
        mode: 'max' or 'min'
    
    Returns:
        Dictionary with extremum value and frequency
    """
    param = getattr(network, parameter)
    
    if mode == 'max':
        idx = np.argmax(np.abs(param))
        value = param[idx]
    else:
        idx = np.argmin(np.abs(param))
        value = param[idx]
    
    freq = network.frequency.f[idx]
    
    return {
        'value': value,
        'frequency': freq,
        'magnitude': np.abs(value),
        'phase_deg': np.angle(value, deg=True),
        'index': idx
    }


def analyze_frequency_response(network, parameter='s11'):
    """
    Analyze frequency response of network parameter.
    
    Args:
        network: Network object
        parameter: Parameter name
    
    Returns:
        Dictionary with frequency response analysis
    """
    param = getattr(network, parameter)
    param_mag = np.abs(param)
    param_deg = np.angle(param, deg=True)
    
    analysis = {
        'min_mag': np.min(param_mag),
        'max_mag': np.max(param_mag),
        'mean_mag': np.mean(param_mag),
        'std_mag': np.std(param_mag),
        'min_phase': np.min(param_deg),
        'max_phase': np.max(param_deg),
        'mean_phase': np.mean(param_deg),
        'std_phase': np.std(param_deg)
    }
    
    return analysis


def compare_networks(network1, network2, parameter='s11'):
    """
    Compare two networks parameter by parameter.
    
    Args:
        network1: First network
        network2: Second network
        parameter: Parameter to compare
    
    Returns:
        Dictionary with comparison results
    """
    param1 = getattr(network1, parameter)
    param2 = getattr(network2, parameter)
    
    # Calculate difference
    diff = param1 - param2
    ratio = param1 / param2
    
    comparison = {
        'difference': diff,
        'ratio': ratio,
        'max_diff_mag': np.max(np.abs(diff)),
        'mean_diff_mag': np.mean(np.abs(diff)),
        'max_ratio_mag': np.max(np.abs(ratio)),
        'mean_ratio_mag': np.mean(np.abs(ratio))
    }
    
    return comparison


def plot_network_comparison(network1, network2, parameter='s11', 
                            title='Network Comparison'):
    """
    Plot comparison of two networks.
    
    Args:
        network1: First network
        network2: Second network
        parameter: Parameter to compare
        title: Plot title
    
    Returns:
        matplotlib figure object
    """
    rf.stylely()
    
    param1 = getattr(network1, parameter)
    param2 = getattr(network2, parameter)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Magnitude comparison
    axes[0].plot(network1.frequency.f_scaled, 
                 20*np.log10(np.abs(param1)), 
                 label=network1.name)
    axes[0].plot(network2.frequency.f_scaled, 
                 20*np.log10(np.abs(param2)), 
                 label=network2.name)
    axes[0].set_xlabel('Frequency (GHz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title(f'{parameter.upper()} Magnitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Phase comparison
    axes[1].plot(network1.frequency.f_scaled, 
                 np.angle(param1, deg=True), 
                 label=network1.name)
    axes[1].plot(network2.frequency.f_scaled, 
                 np.angle(param2, deg=True), 
                 label=network2.name)
    axes[1].set_xlabel('Frequency (GHz)')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].set_title(f'{parameter.upper()} Phase')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def analyze_cascaded_response(networks, parameter='s21'):
    """
    Analyze response of cascaded networks.
    
    Args:
        networks: List of networks
        parameter: Parameter to analyze
    
    Returns:
        Dictionary with cascaded response analysis
    """
    # Calculate cascaded response
    cascaded = networks[0]
    for network in networks[1:]:
        cascaded = cascaded ** network
    
    # Analyze individual and cascaded responses
    analysis = {
        'individual': [getattr(network, parameter) for network in networks],
        'cascaded': getattr(cascaded, parameter)
    }
    
    return analysis


def generate_network_report(network, output_file=None):
    """
    Generate comprehensive network analysis report.
    
    Args:
        network: Network object
        output_file: Output file path (optional)
    
    Returns:
        Report string
    """
    # Analyze network properties
    properties = analyze_network_properties(network)
    
    # Analyze S-parameters
    s_params = extract_s_parameters(network)
    
    # Calculate derived parameters
    return_loss = calculate_return_loss(network)
    insertion_loss = calculate_insertion_loss(network)
    vswr = calculate_vswr(network)
    
    # Find extremum values
    s11_max = find_extremum_values(network, 's11', 'max')
    s21_max = find_extremum_values(network, 's21', 'max')
    
    # Generate report
    report = []
    report.append("=" * 60)
    report.append(f"Network Analysis Report: {network.name}")
    report.append("=" * 60)
    report.append()
    
    report.append("Basic Properties:")
    report.append(f"  Number of ports: {properties['nports']}")
    report.append(f"  Number of frequency points: {properties['nfreq']}")
    report.append(f"  Frequency range: {properties['frequency_range'][0]:.3f} - "
                     f"{properties['frequency_range'][1]:.3f} GHz")
    report.append(f"  Is reciprocal: {properties['is_reciprocal']}")
    report.append(f"  Is symmetric: {properties['is_symmetric']}")
    report.append(f"  Is lossless: {properties['is_lossless']}")
    report.append()
    
    report.append("S-Parameter Extremum Values:")
    report.append(f"  S11 max: {s11_max['magnitude']:.6f} at "
                 f"{s11_max['frequency']:.3f} GHz")
    report.append(f"  S21 max: {s21_max['magnitude']:.6f} at "
                 f"{s21_max['frequency']:.3f} GHz")
    report.append()
    
    report.append("Derived Parameters:")
    report.append(f"  Max return loss: {np.max(return_loss):.3f} dB")
    report.append(f"  Min insertion loss: {np.min(insertion_loss):.3f} dB")
    report.append(f"  Max VSWR: {np.max(vswr):.3f} dB")
    report.append()
    
    report = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    
    return report


def main():
    """Example usage of network analysis functions."""
    print("scikit-rf Network Analysis Tools")
    print("=" * 40)
    print()
    print("Available functions:")
    print("- analyze_network_properties: Analyze basic network properties")
    print("- extract_s_parameters: Extract S-parameters")
    print("- extract_z_parameters: Extract Z-parameters")
    print("- calculate_return_loss: Calculate return loss")
    print("- calculate_insertion_loss: Calculate insertion loss")
    print("- calculate_vswr: Calculate VSWR")
    print("- find_extremum_values: Find extremum values")
    print("- analyze_frequency_response: Analyze frequency response")
    print("- compare_networks: Compare two networks")
    print("- plot_network_comparison: Plot network comparison")
    print("- analyze_cascaded_response: Analyze cascaded response")
    print("- generate_network_report: Generate comprehensive report")
    print()
    print("Example usage:")
    print("  import skrf as rf")
    print("  from network_analysis import analyze_network_properties, "
          "generate_network_report")
    print("  ")
    print("  # Load network")
    print("  network = rf.Network('data/device.s2p')")
    print("  ")
    print("  # Analyze properties")
    print("  properties = analyze_network_properties(network)")
    print("  print(properties)")
    print("  ")
    print("  # Generate report")
    print("  report = generate_network_report(network, 'report.txt')")
    print("  print(report)")


if __name__ == "__main__":
    main()
