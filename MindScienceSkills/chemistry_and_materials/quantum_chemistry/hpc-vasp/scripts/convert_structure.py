#!/usr/bin/env python3
"""
Read local structure files and convert to standard formats

Supported formats: .cif, .xyz, .vasp/POSCAR, .xsd, .xtd, etc.

Usage:
    python convert_structure.py --input my_structure.cif
    python convert_structure.py --input my_structure.cif --output ./converted

 Dependency: pymatgen>=2023.3.10
 Install: pip install pymatgen
 """

import argparse
import sys
import os

try:
    from pymatgen.core import Structure
except ImportError:
    print("Error: Missing required dependency pymatgen. Please run: pip install pymatgen", file=sys.stderr)
    sys.exit(1)

# Supported input formats
INPUT_FORMATS = {
    '.cif', '.xyz', '.vasp', '.poscar', '.xsd', '.xtd',
    '.in', '.ins', '.car', '.pdb', '.mol', '.sdf'
}
# Supported output formats
OUTPUT_FORMATS = ["cif", "poscar", "vasp", "xyz", "json"]


def validate_input_file(input_file: str) -> bool:
    """Validate if input file is valid"""
    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}", file=sys.stderr)
        return False

    ext = os.path.splitext(input_file)[1].lower()
    if ext not in INPUT_FORMATS:
        print(f"Error: Unsupported input format '{ext}'", file=sys.stderr)
        print(f"Supported formats: {', '.join(sorted(INPUT_FORMATS))}", file=sys.stderr)
        return False

    return True


def convert_structure(input_file: str, output_dir: str = ".", output_formats: list = None) -> list:
    """Convert structure file to specified formats"""
    if output_formats is None:
        output_formats = ["cif", "poscar"]

    if not validate_input_file(input_file):
        sys.exit(1)

    try:
        structure = Structure.from_file(input_file)
        print(f"Successfully read structure: {structure.composition}")
    except Exception as e:
        print(f"Error: Failed to read file: {e}", file=sys.stderr)
        sys.exit(1)

    # Check minimum atomic distance
    min_dist = structure.distance_matrix.min()
    print(f"Minimum atomic distance: {min_dist:.3f} Å")
    if min_dist < 0.5:
        print("Warning: Atomic distance may be too close, possible structural issues!")

    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    output_files = []
    for fmt in output_formats:
        if fmt not in OUTPUT_FORMATS:
            print(f"Warning: Unsupported format {fmt}, skipping", file=sys.stderr)
            continue

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if fmt == "poscar" or fmt == "vasp":
            output_file = os.path.join(output_dir, f"{base_name}.vasp")
            structure.to(filename=output_file, fmt="poscar")
        elif fmt == "cif":
            output_file = os.path.join(output_dir, f"{base_name}.cif")
            structure.to(filename=output_file, fmt="cif")
        else:
            output_file = os.path.join(output_dir, f"{base_name}.{fmt}")
            try:
                structure.to(filename=output_file, fmt=fmt)
            except Exception as e:
                print(f"Warning: Failed to save {fmt} format: {e}", file=sys.stderr)
                continue

        output_files.append(output_file)
        print(f"Saved: {output_file}")

    if not output_files:
        print("Error: No output files generated", file=sys.stderr)
        sys.exit(1)

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Read local structure files and convert to standard formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_structure.py --input my_structure.cif
    python convert_structure.py --input structure.xyz --formats cif poscar
        """
    )
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", default=".", help="Output directory (default: current directory)")
    parser.add_argument("--formats", nargs="+", choices=OUTPUT_FORMATS,
                        default=["cif", "poscar"], help="Output formats (default: cif poscar)")

    args = parser.parse_args()
    convert_structure(args.input, args.output, args.formats)


if __name__ == "__main__":
    main()
