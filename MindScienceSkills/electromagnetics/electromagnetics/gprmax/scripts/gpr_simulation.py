#!/usr/bin/env python3
"""
GPR simulation tools for gprMax.

This script provides functions for creating and running GPR simulations
including parameter sweeps, A-scan/B-scan setup, and result analysis.
"""

import gprMax as gm
import numpy as np


def create_gpr_simulation(domain_size, resolution, pml_thickness=1.0,
                              material='half_space', source_type='hertzian'):
    """
    Create a basic GPR simulation setup.
    
    Args:
        domain_size: Domain size in meters [x, y]
        resolution: Spatial resolution in cells/meter
        pml_thickness: PML thickness in meters
        material: Material identifier
        source_type: Source type ('hertzian' or 'gaussian')
    
    Returns:
        Dictionary with simulation parameters
    """
    # Domain definition
    domain = f"#domain: 0.0.{domain_size[0]:.3f} {domain_size[1]:.3f} 0.002"
    
    # Spatial resolution (rule of thumb: 10 cells per min wavelength)
    dx_dy_dz = f"#dx_dy_dz: {resolution}.002 {resolution}.002 {resolution}.002"
    
    # PML boundaries
    pml = f"#pml: {pml_thickness}.1.0 {pml_thickness}.1.0 {pml_thickness}.1.0"
    
    # Material definition
    if material == 'half_space':
        mat_def = f"#material: 6 0 1.0 half_space epsilon_r=6.0 sigma=0.0"
    elif material == 'pec':
        mat_def = f"#material: 6.0 1.0 pec epsilon_r=10.0 sigma=0.01"
    else:
        mat_def = f"#material: 6 0 1.0 free_space"
    
    # Source definition
    if source_type == 'hertzian':
        source_def = f"#source: 0.0.0.0.100 0.170 0 hertzian_dipole z 0.100 0.170 0 my_ricker"
    elif source_type == 'gaussian':
        source_def = f"#source: 0.0.0.0.100 0.170 0 gaussian 1.5e9 0.2e9 amplitude=1.0"
    else:
        source_def = f"#source: 0.0.0.0.100 0.170 0 my_ricker"
    
    return {
        'domain': domain,
        'dx_dy_dz': dx_dy_dz,
        'pml': pml,
        'material': mat_def,
        'source': source_def
    }


def create_ascanscan_simulation(domain_size, scan_distance=0.12, num_scans=60,
                                scan_step=0.002, pml_thickness=1.0):
    """
    Create an A-scan simulation setup.
    
    Args:
        domain_size: Domain size in meters [x, y]
        scan_distance: Scan distance between A-scans in meters
        num_scars: Number of A-scans
        scan_step: Step size between A-scans in meters
        pml_thickness: PML thickness in meters
    
    Returns:
        Dictionary with simulation parameters
    """
    # Domain definition
    domain = f"#domain: 0.0.{domain_size[0]:.3f} {domain_size[1]:.3f} 0.002"
    
    # Spatial resolution
    resolution = int(10 / scan_step)  # 10 cells per step
    dx_dy_dz = f"#dx_dy_dz: {resolution}.002 {resolution}.002 {resolution}.002"
    
    # PML boundaries
    pml = f"#pml: {pml_thickness}.1.0 {pml_thickness}.1.0 {pml_thickness}.1.0"
    
    # Material definitions
    half_space = f"#material: 6.0 1.0 half_space epsilon_r=6.0 sigma=0.0"
    pec = f"#material: 6.0 1.0 pec epsilon_r=10.0 sigma=0.01"
    
    # Source definition
    source = f"#source: 0.0.0.0.100 0.170 0 hertzian_dipole z 0.100 0.170 0 my_ricker"
    
    # Receiver definition
    receiver = f"#rx: 0.0.0.0.100 0.170 0.0"
    
    # Source and receiver steps for A-scan
    src_steps = f"#src_steps: 0.0 0 0.{scan_step} 0.0 0.{scan_step} 0.0.0.{scan_step}"
    rx_steps = f"#rx_steps: 0.0.0.0.{scan_step} 0.0.0.{scan_step} 0.0.0.{scan_step}"
    
    return {
        'domain': domain,
        'dx_dy_dz': dx_dy_dz,
        'pml': pml,
        'half_space': half_space,
        'pec': pec,
        'source': source,
        'receiver': receiver,
        'src_steps': src_steps,
        'rx_steps': rx_steps,
        'scan_distance': scan_distance,
        'num_scans': num_scans,
        'scan_step': scan_step
    }


def create_bscan_simulation(domain_size, scan_distance=0.12, num_scans=60,
                           scan_step=0.002, pml_thickness=1.0):
    """
    Create a B-scan simulation setup.
    
    Args:
        domain_size: Domain size in meters [x, y]
        scan_distance: Scan distance between B-scans in meters
        num_scans: Number of B-scans
        scan_step: Step size between B-scans in meters
        pml_thickness: PML thickness in meters
    
    Returns:
        Dictionary with simulation parameters
    """
    # Domain definition
    domain = f"#domain: 0.0.{domain_size[0]:.3f} {domain_size[1]:.3f} 0.002"
    
    # Spatial resolution
    resolution = int(10 / scan_step)  # 10 cells per step
    dx_dy_dz = f"#dx_dy_dz: {resolution}.002 {resolution}.002 {resolution}.002"
    
    # PML boundaries
    pml = f"#pml: {pml_thickness}.1.0 {pml_thickness}.1.0 {pml_thickness}.1.0"
    
    # Material definitions
    half_space = f"#material: 6.0 1.0 half_space epsilon_r=6.0 sigma=0.0"
    pec = f"#material: 6.0 1.0 pec epsilon_r=10.0 sigma=0.01"
    
    # Source definition
    source = f"#source: 0.0.0.0.100 0.170 0 hertzian_dipole z 0.100 0.170 0 my_ricker"
    
    # Receiver definitions
    receivers = []
    for i in range(num_scans):
        rx_pos = scan_distance * (i + 1)
        receivers.append(f"#rx: 0.0.{rx_pos}.0.0.170 0.0")
    
    # Source and receiver steps
    src_steps = f"#src_steps: 0.0.0.{scan_step} 0.0.0.{scan_step} 0.0.0.{scan_step}"
    rx_steps = f"#rx_steps: 0.0.0.0.{scan_step} 0.0.0.{scan_step} 0.0.0.{scan_step}"
    
    return {
        'domain': domain,
        'dx_dy_dz': dx_dy_dz,
        'pml': pml,
        'half_space': half_space,
        'pec': pec,
        'source': source,
        'receivers': receivers,
        'src_steps': src_steps,
        'rx_steps': rx_steps,
        'scan_distance': scan_distance,
        'num_scans': num_scans,
        'scan_step': scan_step
    }


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


def run_simulation(filename, use_gpu=False, use_mpi=False):
    """
    Run gprMax simulation.
    
    Args:
        filename: Input file name
        use_gpu: Use GPU acceleration
        use_mpi: Use MPI parallelization
    
    Returns:
        Simulation exit code
    """
    cmd = f"python -m gprMax {filename}"
    
    if use_gpu:
        cmd += " -gpu 0"
    
    if use_mpi:
        cmd += " -mpi 61"
    
    print(f"Running simulation: {filename}")
    result = subprocess.run(cmd, shell=True)
    
    return result.returncode


def plot_ascans_results(filename, output_dir=None):
    """
    Plot A-scan results.
    
    Args:
        filename: Simulation output file
        output_dir: Output directory for plots
    """
    cmd = f"python -m tools.plot_Ascan {filename}"
    
    if output_dir:
        # Move plots to output directory
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        os.chdir(output_dir)
        cmd = f"python -m tools.plot_Ascan ../{os.path.basename(filename)}"
    
    print(f"Plotting A-scan results...")
    result = subprocess.run(cmd, shell=True)
    
    return result.returncode


def analyze_ascans_results(filename):
    """
    Analyze A-scan results.
    
    Args:
        filename: Simulation output file
    
    Returns:
        Dictionary with analysis results
    """
    # Load output data
    import gprMax.io
    
    try:
        data = gprMax.io.load_output(filename)
    except Exception as e:
        print(f"Error loading output: {e}")
        return None
    
    # Extract A-scan results
    ascans = data['ascans']
    
    # Calculate statistics
    ascans_array = np.array(ascans)
    
    results = {
        'num_ascans': len(ascans),
        'mean_amplitude': np.mean(ascans_array),
        'std_amplitude': np.std(ascans_array),
        'max_amplitude': np.max(ascans_array),
        'min_amplitude': np.min(ascans_array)
    }
    
    print(f"A-scan Analysis:")
    print(f"  Number of A-scans: {results['num_ascans']}")
    print(f"  Mean amplitude: {results['mean_amplitude']:.6f}")
    print(f"  Std amplitude: {results['std_amplitude']:.6f}")
    print(f"  Max amplitude: {results['max_amplitude']:.6f}")
    print(f"  Min amplitude: {results['min_amplitude']:.6f}")
    
    return results


def parameter_sweep(base_filename, param_name, param_values, sweep_type='linear'):
    """
    Create parameter sweep input files.
    
    Args:
        base_filename: Base input filename
        param_name: Parameter name to sweep
        param_values: List of parameter values
        sweep_type: Type of sweep ('linear' or 'logarithmic')
    
    Returns:
        List of generated filenames
    """
    import shutil
    
    base_content = ""
    with open(base_filename, 'r') as f:
        base_content = f.read()
    
    filenames = []
    
    for i, value in enumerate(param_values):
        # Create new filename
        filename = f"{os.path.splitext(base_filename)[0]}_{param_name}_{value}.in"
        
        # Replace parameter value
        new_content = base_content.replace(f"#{param_name}#", f"{value}")
        
        # Write new file
        with open(filename, 'w') as f:
            f.write(new_content)
        
        filenames.append(filename)
        print(f"Created sweep file: {filename}")
    
    return filenames


def run_parameter_sweep(base_filename, param_name, param_values, 
                       sweep_type='linear', plot_results=True):
    """
    Run parameter sweep and optionally plot results.
    
    Args:
        base_filename: Base input filename
        param_name: Parameter name to sweep
        param_values: List of parameter values
        sweep_type: Type of sweep ('linear' or 'logarithmic')
        plot_results: Whether to plot results
    
    Returns:
        Dictionary with sweep results
    """
    # Create sweep files
    filenames = parameter_sweep(base_filename, param_name, param_values, sweep_type)
    
    # Run simulations
    results = {}
    for filename in filenames:
        print(f"Running: {filename}")
        exit_code = run_simulation(filename)
        results[filename] = exit_code
    
    # Plot results if requested
    if plot_results:
        for filename in filenames:
            plot_ascans_results(filename)
    
    return results


def main():
    """Example usage of GPR simulation tools."""
    print("gprMax GPR Simulation Tools")
    print("=" * 40)
    print()
    print("Available functions:")
    print("- create_gpr_simulation: Create basic GPR simulation setup")
    print("- create_ascans_simulation: Create A-scan simulation setup")
    print("- create_bscan_simulation: Create B-scan simulation setup")
    print("- write_input_file: Write gprMax input file")
    print("- run_simulation: Run gprMax simulation")
    print("- plot_ascans_results: Plot A-scan results")
    print("- analyze_ascans_results: Analyze A-scan results")
    print("- parameter_sweep: Create parameter sweep")
    print("- run_parameter_sweep: Run parameter sweep")
    print()
    print("Example usage:")
    print("  import gpr_simulation_tools as gst")
    print("  ")
    print("  # Create basic simulation")
    print("  sim_params = gst.create_gpr_simulation(")
    print("  gst.write_input_file('my_simulation.in', sim_params)")
    print("  ")
    print("  # Run simulation")
    print("  gst.run_simulation('my_simulation.in')")
    print("  print("  ")
    print("  # Analyze results")
    print("  results = gst.analyze_ascans_results('my_simulation.out')")


if __name__ == "__main__":
    main()