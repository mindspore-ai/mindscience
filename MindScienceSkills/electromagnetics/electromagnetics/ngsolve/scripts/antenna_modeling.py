#!/usr/bin/env python3
"""
Antenna modeling tools for NGsolve.

This script provides functions for creating and analyzing antenna models
including parameter sweeps, radiation pattern analysis, and impedance calculations.
"""

import ngsolve as ns
import numpy as np


def create_dipole_antenna(mesh_radius, frequency=1.0e9, 
                          wire_radius=0.001, length=0.25):
    """
    Create a dipole antenna model.
    
    Args:
        mesh_radius: Mesh radius in meters
        frequency: Operating frequency in Hz
        wire_radius: Wire radius in meters
        length: Wire length in meters
    
    Returns:
        Problem setup dictionary
    """
    # Create spherical mesh
    mesh = ns.Mesh()
    mesh.add_sphere((0, 0, 0), mesh_radius)
    
    # Create function space
    V = ns.FunctionSpace('V', 3)
    V.set_parameter_order('E', 'E', 'H')
    
    # Create wire
    wire = ns.Cylinder(mesh_radius, wire_radius, length, 
                          center=(0, 0, length/2))
    
    # Add wire to mesh
    mesh.add_cylinder(wire)
    
    # Create material
    epsilon = ns.Constant('epsilon_r')
    epsilon.set_value(1.0)
    V.set_material(epsilon)
    
    # Create source
    f = ns.Constant('f')
    f.set_value(frequency)
    V.set_source(f)
    
    # Create problem
    problem = ns.HelmholtzEquation('V', mesh, 
                                             degree=ns.grad(V))
    
    return {
        'mesh': mesh,
        'V': V,
        'problem': problem
    }


def create_microstrip_antenna(width=0.025, length=0.5, 
                              frequency=2.4e9, num_elements=10):
    """
    Create a microstrip antenna model.
    
    Args:
        width: Strip width in meters
        length: Strip length in meters
        frequency: Operating frequency in GHz
        num_elements: Number of elements
    
    Returns:
        Problem setup dictionary
    """
    # Create mesh
    mesh = ns.Mesh()
    
    # Add microstrip elements
    element_width = width / num_elements
    for i in range(num_elements):
        x_pos = -length/2 + i * element_width
        mesh.add_cylinder(element_width/2, length, 
                               center=(x_pos, 0, 0))
    
    # Create function space
    V = ns.FunctionSpace('V', 3)
    V.set_parameter_order('E', 'E', 'H')
    
    # Create material
    epsilon = ns.Constant('epsilon_r')
    epsilon.set_value(2.2)
    V.set_material(epsilon)
    
    # Create source
    f = ns.Constant('f')
    f.set_value(frequency)
    V.set_source(f)
    
    # Create problem
    problem = ns.HelmholtzEquation('V', mesh, 
                                             degree=ns.grad(V))
    
    return {
        'mesh': mesh,
        'V': V,
        'problem': problem
    }


def create_yagi_uda_antenna(width=0.05, length=0.2, 
                            frequency=2.4e9, gain=20.0):
    """
    Create a Yagi-Uda antenna model.
    
    Args:
        width: Antenna width in meters
        length: Antenna length in meters
        frequency: Operating frequency in GHz
        gain: Antenna gain in dB
    
    Returns:
        Problem setup dictionary
    """
    # Create mesh
    mesh = ns.Mesh()
    mesh.add_rect(0, width, 0, length)
    
    # Create function space
    V = ns.FunctionSpace('V', 3)
    V.set_parameter_order('E', 'E', 'H')
    
    # Create material with gain
    epsilon = ns.Constant('epsilon_r')
    epsilon.set_value(2.2)
    epsilon.set_conductivity(gain)
    V.set_material(epsilon)
    
    # Create source
    f = ns.Constant('f')
    f.set_value(frequency)
    V.set_source(f)
    
    # Create problem
    problem = ns.HelmholtzEquation('V', mesh, 
                                             degree=ns.grad(V))
    
    return {
        'mesh': mesh,
        'V': V,
        'problem': problem
    }


def analyze_radiation_pattern(mesh, V, component='Ez'):
    """
    Analyze radiation pattern from field data.
    
    Args:
        mesh: NGsolve mesh
        V: Function space
        component: Field component to analyze
    
    Returns:
        Analysis results dictionary
    """
    # Get field data
    E = V.get_subfunction(component)
    E_array = E.vector().get_array()
    
    # Calculate far-field pattern
    # This would require post-processing
    # See post_processing.py for details
    
    return {
        'field_data': E_array,
        'mesh': mesh,
        'V': V
    }


def calculate_input_impedance(s11_freq, z0_load, 
                                 wire_radius=0.001):
    """
    Calculate input impedance of dipole antenna.
    
    Args:
        s11_freq: S11 frequency at load
        z0_load: Load impedance at load
        wire_radius: Wire radius
    
    Returns:
        Input impedance
    """
    # Approximate input impedance
    # Z_in = 73.13 + 41.25 * log10(s11_freq / z0_load)
    Z_in = 73.13 + 41.25 * np.log10(s11_freq / z0_load)
    
    return Z_in


def calculate_radiation_efficiency(s11_freq, z0_load, 
                                   wire_radius=0.001):
    """
    Calculate radiation efficiency of dipole antenna.
    
    Args:
        s11_freq: S11 frequency at load
        z0_load: Load impedance at load
        wire_radius: Wire radius
    
    Returns:
        Radiation efficiency
    """
    # Approximate radiation efficiency
    # R_rad = 4 * (Z_in**2 / (Z_in + Z_in))
    Z_in = calculate_input_impedance(s11_freq, z0_load, wire_radius)
    R_rad = 4 * (Z_in**2 / (Z_in + Z_in))
    
    return R_rad


def parameter_sweep_frequency(frequencies, base_setup):
    """
    Create parameter sweep over frequencies.
    
    Args:
        frequencies: List of frequencies to sweep
        base_setup: Base problem setup function
    
    Returns:
        Dictionary with sweep results
    """
    results = {}
    
    for freq in frequencies:
        print(f"Simulating at {freq:.2f} GHz")
        sim_setup = base_setup(freq)
        
        # Create solver
        solver = ns.HelmholtzSolver('V', sim_setup['mesh'], 
                                             sim_setup['problem'])
        
        # Solve
        solver.solve()
        
        # Store results
        results[freq] = {
            'mesh': sim_setup['mesh'],
            'V': sim_setup['V'],
            'solver': solver
        }
    
    return results


def plot_radiation_pattern(frequencies, results, component='Ez'):
    """
    Plot radiation pattern vs frequency.
    
    Args:
        frequencies: List of frequencies
        results: Dictionary with sweep results
        component: Field component to plot
    
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    radiation_efficiencies = []
    for freq in frequencies:
        sim_setup = results[freq]
        solver = sim_setup['solver']
        
        # Extract field data
        E = sim_setup['V'].get_subfunction(component)
        E_array = E.vector().get_array()
        
        # Calculate radiation efficiency
        # See calculate_radiation_efficiency()
        
        radiation_efficiencies.append(1.0)  # Placeholder
    
    ax.plot(frequencies, radiation_efficiencies, 'o-', 
            label=f"Radiation Efficiency")
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Radiation Efficiency')
    ax.set_title('Dipole Antenna Radiation Pattern')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def main():
    """Example usage of antenna modeling functions."""
    print("NGsolve Antenna Modeling Tools")
    print("=" * 40)
    print()
    print("Available functions:")
    print("- create_dipole_antenna: Create dipole antenna model")
    print("- create_microstrip_antenna: Create microstrip antenna model")
    print("- create_yagi_uda_antenna: Create Yagi-Uda antenna model")
    print("- analyze_radiation_pattern: Analyze radiation pattern")
    print("- calculate_input_impedance: Calculate input impedance")
    print("- calculate_radiation_efficiency: Calculate radiation efficiency")
    print("- parameter_sweep_frequency: Create frequency sweep")
    print("- plot_radiation_pattern: Plot radiation pattern")
    print()
    print("Example usage:")
    print("  import antenna_modeling as am")
    print("  ")
    print("  # Create dipole antenna")
    print("  sim_params = am.create_dipole_antenna(mesh_radius=0.01, frequency=1.0e9)")
    print("  ")
    print("  # Create solver")
    print("  solver = ns.HelmholtzSolver('V', sim_params['mesh'], ")
    print(" #                                              sim_params['problem'])")
    print("  # Solve")
    print(" solver.solve()")
    print("  ")
    print("  # Analyze results")
    print("  E = sim_params['V'].get_subfunction('E')")
    print("  E_array = E.vector().get_array()")
    print("  print(f"Max E field: {np.max(np.abs(E_array))}")")


if __name__ == "__main__":
    main()