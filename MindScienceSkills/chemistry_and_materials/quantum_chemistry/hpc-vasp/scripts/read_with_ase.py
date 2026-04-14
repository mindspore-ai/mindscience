#!/usr/bin/env python3
"""
Read structure files in multiple formats using ASE

Supported formats: .cif, .xyz, .vasp/POSCAR, .xsd, etc.

Usage:
    python read_with_ase.py --input my_structure.xyz
    python read_with_ase.py --input my_structure.cif --output-dir ./converted

 Dependency: ase
 Install: pip install ase
 """

import argparse
import sys
import os

try:
    from ase.io import read, write
except ImportError:
    print("Error: Missing required dependency ase. Please run: pip install ase", file=sys.stderr)
    sys.exit(1)

OUTPUT_FORMATS = ["cif", "vasp", "poscar", "xyz", "json"]


def read_and_convert(input_file: str, output_dir: str = ".", output_formats: list = None) -> list:
    """Read and convert structure files using ASE"""
    if output_formats is None:
        output_formats = ["cif", "vasp"]

    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}", file=sys.stderr)
        sys.exit(1)

    try:
        atoms = read(input_file)
        print(f"Successfully read structure: {atoms.get_chemical_formula()}")
    except Exception as e:
        print(f"Error: Failed to read file: {e}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_files = []

    for fmt in output_formats:
        if fmt not in OUTPUT_FORMATS:
            print(f"Warning: Unsupported format {fmt}, skipping", file=sys.stderr)
            continue

        output_file = os.path.join(output_dir, f"{base_name}.{fmt}")

        try:
            if fmt == "vasp" or fmt == "poscar":
                write(output_file, atoms, format="vasp")
            else:
                write(output_file, atoms, format=fmt)
            output_files.append(output_file)
            print(f"Saved: {output_file}")
        except Exception as e:
            print(f"Error: Failed to save {fmt} format: {e}", file=sys.stderr)

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Read and convert structure files using ASE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python read_with_ase.py --input my_structure.xyz
    python read_with_ase.py --input structure.cif --formats cif vasp
        """
    )
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output-dir", default=".", help="Output directory (default: current directory)")
    parser.add_argument("--formats", nargs="+", choices=OUTPUT_FORMATS,
                        default=["cif", "vasp"], help="Output formats (default: cif vasp)")

    args = parser.parse_args()
    read_and_convert(args.input, args.output_dir, args.formats)


if __name__ == "__main__":
    main()
