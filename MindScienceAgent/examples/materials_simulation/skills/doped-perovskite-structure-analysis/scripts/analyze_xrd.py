#!/usr/bin/env python
"""
Calculate and plot XRD patterns from crystal structures.

Usage:
    python analyze_xrd.py --input CONTCAR --output xrd_pattern.png
    python analyze_xrd.py --input1 pure.cif --input2 doped.cif --compare
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.vasp import Poscar


def calculate_xrd(structure: Structure, wavelength: str = "CuKa", 
                  two_theta_range: tuple = (10, 90)):
    """Calculate XRD pattern for a structure."""
    xrd_calc = XRDCalculator(wavelength=wavelength)
    pattern = xrd_calc.get_pattern(structure, two_theta_range=two_theta_range)
    return pattern


def plot_single_xrd(pattern, output: str, title: str = "XRD Pattern"):
    """Plot a single XRD pattern."""
    plt.figure(figsize=(10, 6))
    plt.plot(pattern.x, pattern.y, 'b-', linewidth=1.5)
    plt.xlabel('2θ (degrees)', fontsize=14)
    plt.ylabel('Intensity (a.u.)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"XRD pattern saved to: {output}")
    plt.close()


def plot_comparison(patterns, labels, output: str, title: str = "XRD Comparison"):
    """Plot multiple XRD patterns for comparison."""
    plt.figure(figsize=(12, 6))
    colors = ['b', 'r', 'g', 'm', 'c']
    
    for i, (pattern, label) in enumerate(zip(patterns, labels)):
        offset = i * max(patterns[0].y) * 0.1  # Offset for visibility
        plt.plot(pattern.x, pattern.y + offset, 
                color=colors[i % len(colors)], 
                label=label, linewidth=1.5)
    
    plt.xlabel('2θ (degrees)', fontsize=14)
    plt.ylabel('Intensity (a.u.)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"XRD comparison saved to: {output}")
    plt.close()


def print_peaks(pattern, top_n: int = 10):
    """Print main diffraction peaks."""
    print(f"\nTop {top_n} diffraction peaks:")
    print("-" * 60)
    print(f"{'2θ (°)':<12} {'Intensity':<12} {'d-spacing (Å)':<15} {'hkl'}")
    print("-" * 60)
    
    # Sort by intensity
    sorted_indices = np.argsort(pattern.y)[::-1][:top_n]
    
    for i in sorted_indices:
        hkl_str = str(pattern.hkls[i][0]['hkl']) if pattern.hkls[i] else "N/A"
        print(f"{pattern.x[i]:<12.2f} {pattern.y[i]:<12.2f} {pattern.d_hkl[i]:<15.4f} {hkl_str}")


def analyze_peak_shift(pattern1, pattern2, label1: str, label2: str):
    """Analyze peak shifts between two patterns."""
    print(f"\nPeak shift analysis ({label1} vs {label2}):")
    print("-" * 50)
    
    # Compare top peaks
    top_n = min(5, len(pattern1.x), len(pattern2.x))
    
    for i in range(top_n):
        shift = pattern2.x[i] - pattern1.x[i]
        d_change = (pattern2.d_hkl[i] - pattern1.d_hkl[i]) / pattern1.d_hkl[i] * 100
        
        hkl_str = str(pattern1.hkls[i][0]['hkl']) if pattern1.hkls[i] else "N/A"
        print(f"Peak {i+1} ({hkl_str}): Δ2θ = {shift:+.3f}°, Δd = {d_change:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Calculate and analyze XRD patterns")
    parser.add_argument("--input", "-i", help="Input structure file (single mode)")
    parser.add_argument("--input1", help="First structure file (comparison mode)")
    parser.add_argument("--input2", help="Second structure file (comparison mode)")
    parser.add_argument("--output", "-o", default="xrd_pattern.png", help="Output image file")
    parser.add_argument("--wavelength", "-w", default="CuKa", 
                       choices=["CuKa", "MoKa", "CoKa", "FeKa"],
                       help="X-ray wavelength")
    parser.add_argument("--compare", action="store_true", help="Compare two structures")
    parser.add_argument("--title", help="Plot title")
    
    args = parser.parse_args()
    
    if args.compare and args.input1 and args.input2:
        # Comparison mode
        struct1 = Structure.from_file(args.input1)
        struct2 = Structure.from_file(args.input2)
        
        pattern1 = calculate_xrd(struct1, args.wavelength)
        pattern2 = calculate_xrd(struct2, args.wavelength)
        
        labels = [args.input1, args.input2]
        title = args.title or f"XRD Comparison: {args.input1} vs {args.input2}"
        plot_comparison([pattern1, pattern2], labels, args.output, title)
        
        print_peaks(pattern1)
        print_peaks(pattern2)
        analyze_peak_shift(pattern1, pattern2, args.input1, args.input2)
        
    elif args.input:
        # Single structure mode
        try:
            # Try POSCAR format first
            struct = Poscar.from_file(args.input).structure
        except:
            # Fall back to general structure reader
            struct = Structure.from_file(args.input)
        
        pattern = calculate_xrd(struct, args.wavelength)
        title = args.title or f"XRD Pattern: {args.input}"
        plot_single_xrd(pattern, args.output, title)
        print_peaks(pattern)
        
    else:
        parser.error("Either --input or both --input1 and --input2 with --compare are required")


if __name__ == "__main__":
    main()
