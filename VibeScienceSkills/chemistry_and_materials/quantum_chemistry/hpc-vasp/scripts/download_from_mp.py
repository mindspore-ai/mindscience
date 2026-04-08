#!/usr/bin/env python3
"""
Download structure files from Materials Project

Usage:
    python download_from_mp.py --api-key YOUR_API_KEY --material-id mp-149
    python download_from_mp.py --api-key YOUR_API_KEY --formula SiO2

 Dependency: pymatgen>=2023.3.10
 Install: pip install pymatgen
 """

import argparse
import sys
import os
import re

try:
    from pymatgen.ext.matproj import MPRester
    from pymatgen.core import Structure
except ImportError:
    print("Error: Missing required dependency pymatgen. Please run: pip install pymatgen", file=sys.stderr)
    sys.exit(1)

# Materials Project API Key format validation (32-character hex string)
API_KEY_PATTERN = re.compile(r'^[a-f0-9]{32}$', re.IGNORECASE)

# Material ID format validation (mp-number)
MATERIAL_ID_PATTERN = re.compile(r'^mp-\d+$', re.IGNORECASE)


def validate_api_key(api_key: str) -> bool:
    """Validate if API Key format is valid"""
    if not api_key:
        print("Error: API Key cannot be empty", file=sys.stderr)
        return False

    # Remove possible prefix
    api_key_clean = api_key.strip()

    if not API_KEY_PATTERN.match(api_key_clean):
        print("Error: API Key format invalid", file=sys.stderr)
        print("Materials Project API Key should be a 32-character hex string", file=sys.stderr)
        print("Please visit https://materialsproject.org/dashboard to get a valid API Key", file=sys.stderr)
        return False

    return True


def validate_material_id(material_id: str) -> bool:
    """Validate if material ID format is valid"""
    if not MATERIAL_ID_PATTERN.match(material_id.strip()):
        print(f"Error: Material ID format invalid '{material_id}'", file=sys.stderr)
        print("Material ID format should be mp-number, e.g., mp-149, mp-1234", file=sys.stderr)
        return False
    return True


def download_by_material_id(api_key: str, material_id: str, output_dir: str = ".") -> str:
    """Download structure by material ID"""
    if not validate_material_id(material_id):
        sys.exit(1)

    try:
        with MPRester(api_key) as mpr:
            structure = mpr.get_structure_by_material_id(material_id)
        output_file = os.path.join(output_dir, f"{material_id}.cif")
        structure.to(filename=output_file, fmt="cif")
        print(f"Successfully downloaded structure {material_id} to {output_file}")
        return output_file
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "unauthorized" in error_msg:
            print(f"Error: API Key invalid or expired", file=sys.stderr)
            print("Please visit https://materialsproject.org/dashboard to get a new API Key", file=sys.stderr)
        elif "not found" in error_msg or "no structure" in error_msg:
            print(f"Error: Material {material_id} not found", file=sys.stderr)
        else:
            print(f"Error: Failed to download material {material_id}: {e}", file=sys.stderr)
        sys.exit(1)


def download_by_formula(api_key: str, formula: str, output_dir: str = ".") -> str:
    """Download structure by formula (downloads first one)"""
    formula_clean = formula.strip()

    try:
        with MPRester(api_key) as mpr:
            structures = mpr.get_structures(formula_clean)
        if not structures:
            print(f"Error: No structure found for formula {formula_clean}", file=sys.stderr)
            sys.exit(1)
        structure = structures[0]
        output_file = os.path.join(output_dir, f"{formula_clean}.cif")
        structure.to(filename=output_file, fmt="cif")
        print(f"Successfully downloaded structure {formula_clean} to {output_file}")
        return output_file
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "unauthorized" in error_msg:
            print(f"Error: API Key invalid or expired", file=sys.stderr)
            print("Please visit https://materialsproject.org/dashboard to get a new API Key", file=sys.stderr)
        else:
            print(f"Error: Failed to download formula {formula_clean}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download structure files from Materials Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_from_mp.py --api-key 1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d --material-id mp-149
    python download_from_mp.py --api-key 1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d --formula SiO2 --output ./structures
        """
    )
    parser.add_argument("--api-key", required=True, help="Materials Project API Key (32-character hex string)")
    parser.add_argument("--material-id", help="Material ID (e.g., mp-149)")
    parser.add_argument("--formula", help="Chemical formula (e.g., SiO2)")
    parser.add_argument("--output", default=".", help="Output directory (default: current directory)")

    args = parser.parse_args()

    if not args.material_id and not args.formula:
        parser.error("Must specify --material-id or --formula")

    if not validate_api_key(args.api_key):
        sys.exit(1)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.material_id:
        download_by_material_id(args.api_key, args.material_id, args.output)
    else:
        download_by_formula(args.api_key, args.formula, args.output)


if __name__ == "__main__":
    main()
