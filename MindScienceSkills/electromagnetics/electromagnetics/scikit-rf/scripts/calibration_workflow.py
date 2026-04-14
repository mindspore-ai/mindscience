#!/usr/bin/env python3
"""
Complete calibration workflow for scikit-rf.

This script provides a complete calibration workflow including
loading standards, creating calibration, applying to DUTs, and
quality verification.
"""

import skrf as rf
from skrf.calibration import OnePort, SOLT, EightTerm
import matplotlib.pyplot as plt
import numpy as np


def load_calibration_data(ideals_dir, measured_dir, standards):
    """
    Load ideal and measured calibration data.
    
    Args:
        ideals_dir: Directory containing ideal responses
        measured_dir: Directory containing measured responses
        standards: List of standard names
    
    Returns:
        Dictionary with ideal and measured networks
    """
    # Load all ideal responses
    ideals_dict = rf.io.read_all(ideals_dir, contains='s1p')
    
    # Load all measured responses
    measured_dict = rf.io.read_all(measured_dir, contains='s1p')
    
    # Filter by standards
    ideals = [ideals_dict[k] for k in standards if k in ideals_dict]
    measured = [measured_dict[k] for k in standards if k in measured_dict]
    
    return {
        'ideals': ideals,
        'measured': measured,
        'ideals_dict': ideals_dict,
        'measured_dict': measured_dict
    }


def create_one_port_calibration(cal_data, name='one_port_cal'):
    """
    Create one-port calibration.
    
    Args:
        cal_data: Calibration data dictionary
        name: Calibration name
    
    Returns:
        OnePort calibration object
    """
    cal = OnePort(
        ideals=cal_data['ideals'],
        measured=cal_data['measured'],
        name=name
    )
    
    return cal


def create_two_port_calibration(cal_data, name='two_port_cal', 
                              isolation=None, switch_terms=None):
    """
    Create two-port calibration.
    
    Args:
        cal_data: Calibration data dictionary
        name: Calibration name
        isolation: Isolation network (optional)
        switch_terms: Switch terms (optional)
    
    Returns:
        Two-port calibration object
    """
    if switch_terms is not None:
        # Use EightTerm with switch terms
        cal cal = EightTerm(
            ideals=cal_data['ideals'],
            measured=cal_data['measured'],
            switch_terms=switch_terms,
            name=name
        )
    else:
        # Use SOLT
        cal = SOLT(
            ideals=cal_data['ideals'],
            measured=cal_data['measured'],
            isolation=isolation,
            name=name
        )
    
    return cal


def run_calibration(cal):
    """
    Run calibration algorithm.
    
    Args:
        cal: Calibration object
    
    Returns:
        Calibration object after running
    """
    print(f"Running calibration: {cal.name}")
    cal.run()
    print("Calibration complete")
    
    return cal


def apply_calibration(cal, dut_network, output_name=None):
    """
    Apply calibration to DUT.
    
    Args:
        cal: Calibration object
        dut_network: DUT network
        output_name: Output name (optional)
    
    Returns:
        Calibrated DUT network
    """
    print(f"Applying calibration to DUT: {dut_network.name}")
    dut_calibrated = cal.apply_cal(dut_network)
    
    if output_name:
        dut_calibrated.name = output_name
    
    return dut_calibrated


def verify_calibration_quality(cal):
    """
    Verify calibration quality by analyzing residuals.
    
    Args:
        cal: Calibration object
    
    Returns:
        Dictionary with quality metrics
    """
    print("Verifying calibration quality...")
    
    # Get residuals
    residuals = cal.residuals
    
    # Calculate residual statistics
    residual_mag = np.abs(residuals.s_mag)
    max_residual = np.max(residual_mag)
    mean_residual = np.mean(residual_mag)
    std_residual = np.std(residual_mag)
    
    # Get uncertainty
    uncertainty = cal.uncertainty
    max_uncertainty = np.max(np.abs(uncertainty.s_mag))
    
    quality_metrics = {
        'max_residual': max_residual,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'max_uncertainty': max_uncertainty
    }
    
    print(f"  Max residual: {max_residual:.6f}")
    print(f"  Mean residual: {mean_residual:.6f}")
    print(f"  Std residual: {std_residual:.6f}")
    print(f"  Max uncertainty: {max_uncertainty:.6f}")
    
    return quality_metrics


def plot_calibration_results(cal, save_path=None):
    """
    Plot calibration results.
    
    Args:
        cal: Calibration object
        save_path: Path to save plot (optional)
    
    Returns:
        matplotlib figure object
    """
    print("Plotting calibration results...")
    
    # Apply skrf plotting style
    rf.stylely()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot residuals
    cal.residuals.s11.plot_s_db(ax=axes[0, 0], m=1, n=0)
    axes[0, 0].set_title('S11 Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    cal.residuals.s21.plot_s_db(ax=axes[0, 1], m=1, n=0)
    axes[0, 1].set_title('S21 Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot uncertainty
    cal.uncertainty.plot_s_db(ax=axes[1, 0], m=1, n=0)
    axes[1, 0].set_title('Uncertainty')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot error coefficients
    cal.coef.plot_s_db(ax=axes[1, 1], m=1, n=0)
    axes[1, 1].set_title('Error Coefficients')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def save_calibration(cal, file_path):
    """
    Save calibration to file.
    
    Args:
        cal: Calibration object
        file_path: Output file path
    """
    print(f"Saving calibration to {file_path}")
    cal.write(file_path)
    print("Calibration saved")


def load_calibration(file_path):
    """
    Load calibration from file.
    
    Args:
        file_path: Input file path
    
    Returns:
        Calibration object
    """
    print(f"Loading calibration from {file_path}")
    cal = rf.io.read(file_path)
    print("Calibration loaded")
    
    return cal


def complete_calibration_workflow(ideals_dir, measured_dir, duts_dir, 
                               standards, cal_type='one_port'):
    """
    Complete calibration workflow.
    
    Args:
        ideals_dir: Directory containing ideal responses
        measured_dir: Directory containing measured responses
        duts_dir: Directory containing DUTs
        standards: List of standard names
        cal_type: Type of calibration ('one_port' or 'two_port')
    
    Returns:
        Dictionary with calibration results
    """
    print("=" * 60)
    print("Complete Calibration Workflow")
    print("=" * 60)
    print()
    
    # Step 1: Load calibration data
    print("Step 1: Loading calibration data...")
    cal_data = load_calibration_data(ideals_dir, measured_dir, standards)
    print(f"  Loaded {len(cal_data['ideals'])} ideal responses")
    print(f"  Loaded {len(cal_data['measured'])} measured responses")
    print()
    
    # Step 2: Create calibration
    print("Step 2: Creating calibration...")
    if cal cal_type == 'one_port':
        cal = create_one_port_calibration(cal_data)
    else:
        cal = create_two_port_calibration(cal_data)
    print()
    
    # Step 3: Run calibration
    print("Step 3: Running calibration...")
    cal = run_calibration(cal)
    print()
    
    # Step 4: Verify calibration quality
    print("Step 4: Verifying calibration quality...")
    quality = verify_calibration_quality(cal)
    print()
    
    # Step 5: Plot calibration results
    print("Step 5: Plotting calibration results...")
    fig = plot_calibration_results(cal)
    plt.show()
    print()
    
    # Step 6: Apply calibration to DUTs
    print("Step 6: Applying calibration to DUTs...")
    duts = rf.io.read_all(duts_dir, contains='s1p')
    calibrated_duts = {}
    
    for name, dut in duts.items():
        dut_calibrated = apply_calibration(cal, dut, 
                                         output_name=f"{name}_calibrated")
        calibrated_duts[name] = dut_calibrated
        print(f"  Calibrated: {name}")
    
    print()
    
    # Step 7: Save calibrated DUTs
    print("Step 7: Saving calibrated DUTs...")
    for name, dut in calibrated_duts.items():
        output_path = f"{duts_dir}/calibrated/{name}.s2p"
        dut.write_touchstone(output_path)
        print(f"  Saved: {output_path}")
    
    print()
    
    # Step 8: Save calibration
    print("Step 8: Saving calibration...")
    cal_path = f"{ideals_dir}/../calibration.cal"
    save_calibration(cal, cal_path)
    print()
    
    print("=" * 60)
    print("Calibration workflow complete")
    print("=" * 60)
    
    return {
        'calibration': cal,
        'calibrated_duts': calibrated_duts,
        'quality_metrics': quality
    }


def main():
    """Example usage of calibration workflow."""
    print("scikit-rf Calibration Workflow")
    print("=" * 40)
    print()
    print("Available functions:")
    print("- load_calibration_data: Load calibration data")
    print("- create_one_port_calibration: Create one-port calibration")
    print("- create_two_port_calibration: Create two-port calibration")
    print("- run_calibration: Run calibration algorithm")
    print("- apply_calibration: Apply calibration to DUT")
    print("- verify_calibration_quality: Verify calibration quality")
    print("- plot_calibration_results: Plot calibration results")
    print("- save_calibration: Save calibration to file")
    print("- load_calibration: Load calibration from file")
    print("- complete_calibration_workflow: Complete calibration workflow")
    print()
    print("Example usage:")
    print("  import skrf as rf")
    print("  from calibration_workflow import complete_calibration_workflow")
    print("  ")
    print("  # Define standards")
    print("  standards = ['short', 'open', 'load']")
    print("  ")
    print("  # Run complete workflow")
    print("  results = complete_calibration_workflow(")
    print("      ideals_dir='ideals/',")
    print("      measured_dir='measured/',")
    print("      duts_dir='duts/',")
    print("      standards=standards,")
    print("      cal_type='one_port")
    print("  )")


if __name__ == "__main__":
    main()