# Basic simulation setup for gprMax

This script provides functions for setting up basic gprMax simulations
including domain definition, material creation, and source configuration.
"""

import gprMax as gm
import numpy as np


def create_basic_2d_simulation(domain_size, resolution, pml_thickness=1.0):
    """
    Create a basic 2D simulation setup.
    
    Args:
        domain_size: Domain size in meters [x, y]
        resolution: Spatial resolution in cells/meter
        pml_thickness: PML thickness in meters
    
    Returns:
        Dictionary with simulation parameters
    """
    # Domain definition
    domain = [0.0, domain_size[0], 0.0]
    
    # Spatial resolution (meters per cell)
    dx_dy_dz = [resolution, resolution, 1.0]
    
    # PML boundaries
    pml = [pml_thickness, pml_thickness, 0.0]
    
    return {
        'domain': domain,
        'dx_dy_dz': dx_dy_dz,
        'pml': pml
    }


def create_materials(epsilon_r=1.0, sigma=0.0):
    """
    Create basic material definitions.
    
    Args:
        epsilon_r: Relative permittivity
        sigma: Conductivity (S/m)
    
    Returns:
        Dictionary with material definitions
    """
    # Free space (air)
    free_space = f"6 0 1 0 free_space"
    
    # Dielectric half-space
    half_space = f"6 0 1 0 half_space epsilon_r={epsilon_r} sigma={sigma}"
    
    # Perfect electric conductor
    pec = f"6 0 1 0 pec"
    
    return {
        'free_space': free_space,
        'half_space': half_space,
        'pec': pec
    }


def create_hertzian_dipole_source(f_center=1.5e9, amplitude=1.0, 
                                  polarization='z', center=[0, 0, 0.1]):
    """
    Create a Hertzian dipole source.
    
    Args:
        f_center: Center frequency in GHz
        amplitude: Source amplitude
        polarization: Field polarization ('x', 'y', or 'z')
        center: Source position [x, y, z] in meters
    
    Returns:
        Source definition string
    """
    # Create Ricker waveform
    wave_type = 'my_ricker'
    
    # Hertzian dipole source
    source = (f"#waveform: ricker 1.5e9 {amplitude} "
             f"polarization={polarization} "
             f"center={center[0]} {center[1]} {center[2]}")
    
    return source


def create_gaussian_pulse_source(f_center=1.5e9, f_width=0.2e9, amplitude=1.0,
                                center=[0, 0, 0.1]):
    """
    Create a Gaussian pulse source.
    
    Args:
        f_center: Center frequency in GHz
        f_width: Frequency width in GHz
        amplitude: Source amplitude
        center: Source position [x, y, z] in meters
    
    Returns:
        Source definition string
    """
    # Create Gaussian pulse
    source = (f"#waveform: gaussian {f_center} {f_width} {amplitude} "
             f"center={center[0]} {center[1]} {center[2]}")
    
    return source


def create_geometry_objects(material, geometry_type='box', size=None, center=None):
    """
    Create geometry object definitions.
    
    Args:
        material: Material identifier
        geometry_type: Type of geometry ('box', 'cylinder', 'sphere')
        size: Object size [x, y, z] in meters
        center: Object center [x, y, z] in meters
    
    Returns:
        Geometry definition string
    """
    if size is None:
        if geometry_type == 'box':
            size = [0.120, 0.080, 0.0]
        elif geometry_type == 'cylinder':
            size = [0.040, 0.040, 0.10]
        elif geometry_type == 'sphere':
            size = [0.050, 0.050, 0.050]
    
    if center is None:
        center = [0.0, 0.0, 0.0]
    
    if geometry_type == 'box':
        geometry = (f"#box: {size[0]} {size[1]} {size[2]} "
                      f"center={center[0]} {center[1]} {center[2]} "
                      f"material={material}")
    elif geometry_type == 'cylinder':
        geometry = (f"#cylinder: radius={size[0]/2 "
                      f"center={center[0]} {center[1]} {center[2]} "
                      f"height={size[2]} "
                      f"material={material}")
    elif geometry_type == 'sphere':
        geometry = (f"#sphere: radius={size[0] "
                    f"center={center[0]} {center[1]} {center[2]} "
                    f"material={material}")
    
    return geometry


def calculate_minimum_wavelength(frequency, epsilon_r):
    """
    Calculate minimum wavelength in material.
    
    Args:
        frequency: Frequency in GHz
        epsilon_r: Relative permittivity
    
    Returns:
        Minimum wavelength in meters
    """
    # c = 3e8 m/s
    # lambda_min = c / (frequency * sqrt(epsilon_r))
    lambda_min = 299.792458 / (frequency * np.sqrt(epsilon_r))
    return lambda_min


def calculate_spatial_resolution(frequency, epsilon_r, cells_per_wavelength=10):
    """
    Calculate spatial resolution.
    
    Args:
        frequency: Frequency in GHz
        epsilon_r: Relative permittivity
        cells_per_wavelength: Cells per minimum wavelength
    
    Returns:
        Spatial resolution in cells/meter
    """
    lambda_min = calculate_minimum_wavelength(frequency, epsilon_r)
    resolution = cells_per_wavelength / lambda_min
    return resolution


def calculate_time_window(frequency, source_duration=5e-9):
    """
    Calculate appropriate time window.
    
    Args:
        frequency: Frequency in GHz
        source_duration: Source duration in seconds
    
    Returns:
        Time window in seconds
    """
    # t_window = source_duration / (1 / frequency)
    t_window = source_duration * frequency
    return t_window


def create_basic_input_file(filename, domain_size, resolution, pml_thickness=1.0,
                             material='half_space', geometry_type='box'):
    """
    Create a basic gprMax input file.
    
    Args:
        filename: Output filename
        domain_size: Domain size in meters [x, y]
        resolution: Spatial resolution in cells/meter
        pml_thickness: PML thickness in meters
        material: Material identifier
        geometry_type: Type of geometry
    
    Returns:
        Input file content string
    """
    # Get simulation parameters
    sim_params = create_basic_2d_simulation(domain_size, resolution, pml_thickness)
    materials = create_materials()
    
    # Create source
    source = create_hertzian_dipole_source()
    
    # Create geometry
    geometry = create_geometry_objects(material, geometry_type)
    
    # Build input file
    input_file = f"""# {filename}
# Basic 2D simulation

# Domain definition
#domain: {sim_params['domain'][0]} {sim_params['domain'][1]} {sim_params['domain'][2]}

# Spatial resolution (meters)
#dx_dy_dz: {sim_params['dx_dy_dz'][0]} {sim_params['dx_dy_dz'][1]} {sim_params['dx_dy_dz'][2]}

# PML boundaries
#pml: {sim_params['pml'][0]} {sim_params['pml'][1]} {sim_params['pml'][2]}

# Materials
{materials['free_space']}
{materials['half_space']}
{materials['pec']}

# Source
{source}

# Geometry
{geometry}

# Time window
#time_window: 3e-9
"""
    
    return input_file


def create_gpr_ascanscan_input_file(filename, scan_distance=0.12, 
                                 num_scans=60, scan_step=0.002):
    """
    Create a GPR A-scan input file.
    
    Args:
        filename: Output filename
        scan_distance: Scan distance in meters
        num_scans: Number of A-scans
        scan_step: Scan step in meters
    
    Returns:
        Input file content string
    """
    input_file = f"""# {filename}
# GPR A-scan simulation

# Domain definition
#domain: 0.240 0.210 0.002 0.002

# Spatial resolution (meters)
#dx_dy_dz: 0.002 0.002 0.002

# PML boundaries
#pml: 1.0 1.0 1.0 1.0

# Materials
#material: 6 0 1 0 free_space
#material: 6 0 1 0 half_space epsilon_r=6.0 sigma=0.0

# Source
#waveform: ricker 1.5e9 1.0 my_ricker
#hertzian_dipole: z 0.100 0.170 0 my_ricker

# Geometry
#box: 0.120 0.080 0.120 0.080 0.002 pec

# Source and receiver for A-scan
#source: 0.0.0 0.100 0.170 0 hertzian_dipole z 0.100 0.170 0 my_ricker
#rx: 0.0.0 0.080 0.170 0

# Source and receiver for B-scan
#source: 0.0.0 0.100 0.170 0 hertzian_dipole z 0.100 0.170 0 my_ricker
#rx: 0.0.0 0.0.080 0.170 0 hertzian_dipole z 0.0.0.170 0 my_ricker

# Source steps for A-scan
#src_steps: 0.002 0
#rx_steps: 0.002 0

# Time window
#time_window: 3e-9
"""
    
    return input_file


def create_bscan_input_file(filename, scan_distance=0.12, num_scans=60):
    """
    Create a B-scan input file.
    
    Args:
        filename: Output filename
        scan_distance: Scan distance in meters
        num_scans: Number of B-scans
    
    Returns:
        Input file content string
    """
    input_file = f"""# {filename}
# B-scan simulation

# Domain definition
#domain: 0.240 0.210 0.002 0.002

# Spatial resolution (meters)
#dx_dy_dz: 0.002 0.002 0.002

# PML boundaries
#pml: 1.0 1.0 1.0 1.0

# Materials
#material: 6 0 1 0 free_space
#material: 6 0 1 0 half_space epsilon_r=6.0 sigma=0.0

# Source and receiver
#source: 0.0.0 0.100 0.170 0 hertzian_dipole z 0.100 0.170 0 my_ricker
#rx: 0.0.0 0.0.080 0.170 0 hertzian_dipole z 0.0.0.170 0 my_ricker

# Source steps for A-scan
#src_steps: 0.002 0
#rx_steps: 0.002 0

# Time window
#time_window: 3e-9
"""
    
    return input_file


def write_input_file(filename, content):
    """
    Write gprMax input file.
    
    Args:
        filename: Output filename
        content: Input file content string
    """
    with open(filename, 'w') as f:
        f.write(content)
    print(f"Wrote input file: {filename}")


def main():
    """Example usage of basic simulation setup functions."""
    print("gprMax Basic Simulation Setup")
    print("=" * 40)
    print()
    print("Available functions:")
    print("- create_basic_2d_simulation: Create basic 2D simulation setup")
    print("- create_materials: Create material definitions")
    print("- create_hertzian_dipole_source: Create Hertzian dipole source")
    print("- create_gaussian_pulse_source: Create Gaussian pulse source")
    print("- create_geometry_objects: Create geometry objects")
    print("- calculate_minimum_wavelength: Calculate minimum wavelength")
    print("- calculate_spatial_resolution: Calculate spatial resolution")
    print("- calculate_time_window: Calculate time window")
    print("- create_basic_input_file: Create basic input file")
    print("- create_gpr_ascanscan_input_file: Create GPR A-scan input file")
    print("- create_bscan_input_file: Create B-scan input file")
    print("- write_input_file: Write input file")
    print()
    print("Example usage:")
    print("  import basic_simulation")
    print("  ")
    print("  # Create simulation parameters")
    print("  sim_params = basic_simulation.create_basic_2d_simulation(")
    print("  ")
    print("  # Calculate resolution")
    print("  resolution = basic_simulation.calculate_spatial_resolution(1.5e9, 6.0)")
    print("  ")
    print("  # Create input file")
    print("  input_content = basic_simulation.create_basic_input_file('my_simulation.in')")
    print("  basic_simulation.write_input_file('my_simulation.in', input_content)")
    print(" ")
    print("  # Run simulation")
    print(" python -m gprMax my_simulation.in")


if __name__ == "__main__":
    main()