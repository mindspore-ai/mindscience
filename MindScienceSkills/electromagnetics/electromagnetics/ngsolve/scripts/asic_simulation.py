#!/usr/bin/env python3
"""
Basic simulation setup for NGsolve.

This script provides functions for setting up basic NGsolve simulations
including mesh creation, material definition, and problem configuration.
"""

import ngsolve as ns


def create_basic_mesh(domain_size, resolution, element_type='tetrahedron'):
    """
    Create a basic mesh for NGsolve.
    
    Args:
        domain_size: Domain size in meters [x, y, z]
        resolution: Spatial resolution in elements/meter
        element_type: Element type ('tetrahedron', 'triangle')
    
    Returns:
        NGsolve mesh object
    """
    # Create mesh
    mesh = ns.Mesh()
    
    if element_type == 'tetrahedron':
        # Add structured mesh
        mesh.add_rect(0, domain_size[0], domain_size[1], 0, domain_size[2])
    elif element_type == 'triangle':
        # Add unstructured mesh
        mesh.add_triangle([(0, 0, 0),
                               (domain_size[0], domain_size[1], 0),
                               (domain_size[0], domain_size[1], domain_size[2])])
    
    return mesh


def create_function_space(field_type='scalar', dimension=3):
    """
    Create function space for NGsolve.
    
    Args:
        field_type: Type of field ('scalar', 'vector', 'mixed')
        dimension: Number of dimensions (1, 2, or 3)
    
    Returns:
        NGsolve function space object
    """
    if field_type == 'scalar':
        V = ns.FunctionSpace('V')
    elif field_type == 'vector':
        V = ns.VectorFunctionSpace('V', dimension)
    elif field_type == 'mixed':
        V = ns.MixedFunctionSpace('V', 
                                        {'E': ns.VectorValue(dimension), 
                                         'H': ns.VectorValue(dimension)})
    
    return V


def create_constant_material(material_name='epsilon', value=1.0):
    """
    Create constant material for NGsolve.
    
    Args:
        material_name: Material name ('epsilon', 'mu', 'sigma', 'epsilon_r')
        value: Material value
    
    Returns:
        Material definition string
    """
    if material_name == 'epsilon':
        return f"#material: {material_name} {value}"
    elif material_name == 'mu':
        return f"#material: {material_name} {value}"
    elif material_name == 'sigma':
        return f"#material: {material_name} {value}"
    elif material_name == 'epsilon_r':
        return f"#material: {material_name} {value}"
    else:
        return f"#material: {material_name} {value}"


def create_helmholtz_problem(mesh, frequency=1.0e9, polarization='z'):
    """
    Create Helmholtz problem setup.
    
    Args:
        mesh: NGsolve mesh object
        frequency: Source frequency in Hz
        polarization: Field polarization ('x', 'y', 'z')
    
    Returns:
        Problem setup dictionary
    """
    # Create function space
    V = ns.FunctionSpace('V', 3)
    V.set_parameter_order('E', 'E', 'H')
    
    # Set source
    f = ns.Constant('f')
    f.set_value(frequency)
    V.set_source(f)
    
    # Create problem
    problem = ns.HelmholtzEquation('V', mesh, 
                                             degree=ns.grad(V))
    
    return {
        'function_space': V,
        'problem': problem
    }


def create_wave_problem(mesh, frequency=1.0e9, wave_type='helmholtz'):
    """
    Create wave equation problem setup.
    
    Args:
        mesh: NGsolve mesh object
        frequency: Source frequency in Hz
        wave_type: Type of wave equation ('helmholtz', 'wave')
    
    Returns:
        Problem setup dictionary
    """
    # Create function space
    V = ns.FunctionSpace('V', 3)
    V.set_parameter_order('E', 'E', 'H')
    
    # Set material
    epsilon = ns.Constant('epsilon_r')
    epsilon.set_value(4.0)
    V.set_material(epsilon)
    
    # Set source
    f = ns.Constant('f')
    f.set_value(frequency)
    V.set_source(f)
    
    # Create problem
    if wave_type == 'helmholtz':
        problem = ns.HelmholtzEquation('V', mesh, 
                                             degree=ns.grad(V))
    elif wave_type == 'wave':
        problem = ns.WaveEquation('V', mesh, 
                                       degree=ns.grad(V))
    
    return {
        'function_space': V,
        'problem': problem
    }


def calculate_minimum_wavelength(frequency, epsilon_r=1.0):
    """
    Calculate minimum wavelength in material.
    
    Args:
        frequency: Frequency in Hz
        epsilon_r: Relative permittivity
    
    Returns:
        Minimum wavelength in meters
    """
    # c = 3e8 m/s
    # lambda_min = c / (frequency * sqrt(epsilon_r))
    lambda_min = 299.792458 / (frequency * np.sqrt(epsilon_r))
    return lambda_min


def calculate_spatial_resolution(frequency, epsilon_r=1.0, 
                             cells_per_wavelength=10):
    """
    Calculate spatial resolution.
    
    Args:
        frequency: Frequency in Hz
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
        frequency: Frequency in Hz
        source_duration: Source duration in seconds
    
    Returns:
        Time window in seconds
    """
    # t_window = source_duration / (1 / frequency)
    t_window = source_duration * frequency
    return t_window


def create_basic_input_file(filename, domain_size, resolution, frequency=1.0e9, 
                             polarization='z', problem_type='helmholtz'):
    """
    Create a basic NGsolve input file.
    
    Args:
        filename: Output filename
        domain_size: Domain size in meters [x, y, z]
        resolution: Spatial resolution in cells/meter
        frequency: Source frequency in Hz
        polarization: Field polarization
        problem_type: Type of problem ('helmholtz', 'wave')
    
    Returns:
        Input file content string
    """
    # Calculate resolution
    lambda_min = calculate_minimum_wavelength(frequency, 4.0)
    resolution = calculate_spatial_resolution(frequency, 4.0, 10)
    
    # Create mesh
    mesh = create_basic_mesh(domain_size, resolution)
    
    # Create problem setup
    if problem_type == 'helmholtz':
        setup = create_helmholtz_problem(mesh, frequency, polarization)
    elif problem_type == 'wave':
        setup = create_wave_problem(mesh, frequency)
    
    # Build input file
    input_file = f"""# {filename}
# Basic NGsolve simulation

# Domain definition
#domain: {domain_size[0]:.3f} {domain_size[1]:.3f} {domain_size[2]:.3f}

# Spatial resolution (cells/meter)
#dx_dy_dz: {resolution}.3f {resolution}.3f {resolution}.3f {resolution}.3f}

# Function space
{setup['function_space']}

# Material
{create_constant_material('epsilon', 4.0)}

# Source
{setup['problem']}

# Time window
#time_window: 3e-9
"""
    
    return input_file


def main():
    """Example usage of basic simulation setup functions."""
    print("NGsolve Basic Basic Simulation Setup")
    print("=" * 40)
    print()
    print("Available functions:")
    print("- create_basic_mesh: Create basic mesh")
    print("- create_function_space: Create function space")
    print("- create_constant_material: Create constant material")
    print("- create_helmholtz_problem: Create Helmholtz problem")
    print("- create_wave_problem: Create wave problem")
    print("- calculate_minimum_wavelength: Calculate minimum wavelength")
    print("- calculate_spatial_resolution: Calculate spatial resolution")
    print("- calculate_time_window: Calculate time window")
    print("- create_basic_input_file: Create input file")
    print()
    print("Example usage:")
    print("  import basic_simulation")
    print("  ")
    print("  # Create simulation parameters")
    print("  sim_params = basic_simulation.create_basic_simulation(")
    print("  ")
    print("  # Calculate resolution")
    print("  resolution = basic_simulation.calculate_spatial_resolution(1.0e9, 4.0)")
    print("  ")
    print("  # Create input file")
    print("  input_content = basic_simulation.create_basic_input_file('my_simulation.in')")
    print("  basic_simulation.write_input_file('my_simulation.in', input_content)")


if __name__ == "__main__":
    main()