#!/usr/bin/env python
"""
Download crystal structure from Materials Project and save as CIF file.

Usage:
    python get_structure.py --mp-id mp-5229 --api-key YOUR_API_KEY --output STO.cif
"""

import argparse
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure


def download_structure(mp_id: str, api_key: str, output: str = None) -> Structure:
    """
    Download structure from Materials Project.
    
    Args:
        mp_id: Materials Project ID (e.g., "mp-5229")
        api_key: Your Materials Project API key
        output: Output filename (optional)
    
    Returns:
        Structure object
    """
    with MPRester(api_key) as mpr:
        structure = mpr.get_structure_by_material_id(mp_id)
        data = mpr.get_data(mp_id)[0]
        
        print(f"Material: {data['pretty_formula']}")
        print(f"Space Group: {data['spacegroup']['symbol']} (No. {data['spacegroup']['number']})")
        print(f"Band Gap: {data['band_gap']:.3f} eV")
        print(f"Number of atoms: {len(structure)}")
        
        if output:
            structure.to(filename=output, fmt="cif")
            print(f"Structure saved to: {output}")
        
        return structure


def main():
    parser = argparse.ArgumentParser(description="Download structure from Materials Project")
    parser.add_argument("--mp-id", required=True, help="Materials Project ID (e.g., mp-5229)")
    parser.add_argument("--api-key", required=True, help="Materials Project API key")
    parser.add_argument("--output", "-o", default="structure.cif", help="Output filename")
    
    args = parser.parse_args()
    download_structure(args.mp_id, args.api_key, args.output)


if __name__ == "__main__":
    main()
