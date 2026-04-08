#!/usr/bin/env python
"""
Build supercell and create doping model.

Usage:
    python build_doped_model.py --input STO.cif --supercell 2x2x2 --dopant Ba --site Sr --output Ba_STO.cif
"""

import argparse
import numpy as np
from pymatgen.core import Structure


def build_doped_model(
    input_file: str,
    supercell: tuple,
    dopant: str,
    host_site: str,
    output: str,
    dopant_index: int = 0
) -> Structure:
    """
    Build supercell and create doping model.
    
    Args:
        input_file: Input CIF file
        supercell: Supercell dimensions (e.g., (2, 2, 2))
        dopant: Dopant element symbol
        host_site: Host site element to substitute
        output: Output filename
        dopant_index: Index of host site to substitute (default: 0)
    
    Returns:
        Doped structure
    """
    # Load structure
    structure = Structure.from_file(input_file)
    print(f"Loaded structure: {structure.formula}")
    print(f"Number of atoms: {len(structure)}")
    
    # Build supercell
    supercell_matrix = np.diag(supercell)
    structure_supercell = structure * supercell_matrix
    print(f"\nSupercell {supercell[0]}x{supercell[1]}x{supercell[2]}")
    print(f"Number of atoms in supercell: {len(structure_supercell)}")
    
    # Find host sites
    host_indices = [i for i, site in enumerate(structure_supercell) 
                    if site.species_string == host_site]
    print(f"\nFound {len(host_indices)} {host_site} sites")
    
    if dopant_index >= len(host_indices):
        print(f"Warning: dopant_index {dopant_index} out of range, using 0")
        dopant_index = 0
    
    # Create doped structure
    structure_doped = structure_supercell.copy()
    target_index = host_indices[dopant_index]
    structure_doped[target_index] = dopant, structure_doped[target_index].coords
    
    print(f"\nSubstituted {host_site} at index {target_index} with {dopant}")
    print(f"Doped structure formula: {structure_doped.formula}")
    
    # Save
    structure_doped.to(filename=output, fmt="cif")
    print(f"\nDoped structure saved to: {output}")
    
    # Also save undoped supercell for comparison
    undoped_output = output.replace(".cif", "_undoped_supercell.cif")
    structure_supercell.to(filename=undoped_output, fmt="cif")
    print(f"Undoped supercell saved to: {undoped_output}")
    
    return structure_doped


def main():
    parser = argparse.ArgumentParser(description="Build doped supercell model")
    parser.add_argument("--input", "-i", required=True, help="Input CIF file")
    parser.add_argument("--supercell", "-s", default="2x2x2", 
                       help="Supercell dimensions (e.g., 2x2x2)")
    parser.add_argument("--dopant", "-d", required=True, help="Dopant element")
    parser.add_argument("--site", required=True, help="Host site element to substitute")
    parser.add_argument("--output", "-o", default="doped_structure.cif", help="Output filename")
    parser.add_argument("--index", type=int, default=0, help="Index of site to dope")
    
    args = parser.parse_args()
    
    # Parse supercell dimensions
    supercell = tuple(map(int, args.supercell.split("x")))
    
    build_doped_model(
        args.input, supercell, args.dopant, args.site, args.output, args.index
    )


if __name__ == "__main__":
    main()
