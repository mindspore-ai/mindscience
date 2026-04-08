#!/usr/bin/env python3
"""
Build crystal and surface structures using ASE

Usage:
    # Build bulk crystal
    python build_structure_ase.py --element Si --structure diamond --lattice 5.43

    # Build molecule
    python build_structure_ase.py --molecule H2O

    # Build surface (both formats supported)
    python build_structure_ase.py --element Si --structure diamond --surface 111 --layers 3
    python build_structure_ase.py --element Si --structure diamond --surface 1,1,1 --layers 3

 Dependency: ase>=3.23.0
 Install: pip install ase
 """

import argparse
import sys
import os

try:
    from ase.build import bulk, molecule, surface
    from ase.io import write
except ImportError:
    print("Error: Missing required dependency ase. Please run: pip install ase", file=sys.stderr)
    sys.exit(1)

# ASE supported element list
ASE_ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

# Common crystal structures
VALID_STRUCTURES = ['diamond', 'fcc', 'bcc', 'hcp', 'sc', 'rocksalt', 'zincblende', 'wurtzite']


def validate_element(element: str) -> bool:
    """Validate if element symbol is valid"""
    if element not in ASE_ELEMENTS:
        print(f"Error: Invalid element symbol '{element}'", file=sys.stderr)
        print(f"Please refer to ASE element list for valid symbols", file=sys.stderr)
        return False
    return True


def validate_structure(structure: str) -> bool:
    """Validate if crystal structure is valid"""
    if structure not in VALID_STRUCTURES:
        print(f"Error: Invalid crystal structure '{structure}'", file=sys.stderr)
        print(f"Supported structures: {', '.join(VALID_STRUCTURES)}", file=sys.stderr)
        return False
    return True


def parse_surface_indices(surface_str: str) -> tuple:
    """Parse surface indices, supports '111' or '1,1,1' format"""
    # Clean string
    surface_str = surface_str.strip()

    # If contains comma, split by comma
    if ',' in surface_str:
        parts = surface_str.split(',')
    else:
        # Otherwise group every 3 (e.g., '111' -> (1,1,1))
        if len(surface_str) % 3 != 0:
            raise ValueError(f"Surface indices '{surface_str}' format invalid, supports '111' or '1,1,1'")
        parts = [surface_str[i:i+3] for i in range(0, len(surface_str), 3)]

    if len(parts) != 3:
        raise ValueError(f"Surface indices must be 3 integers, e.g., '111' or '1,1,1', got: {parts}")

    try:
        indices = tuple(int(p) for p in parts)
    except ValueError:
        raise ValueError(f"Surface indices must be integers, e.g., '111' or '1,1,1'")

    # Check if valid Miller indices
    if indices == (0, 0, 0):
        raise ValueError("Surface indices cannot all be 0")

    return indices


def build_bulk(element: str, structure: str, lattice: float = None, output_dir: str = ".") -> str:
    """Build bulk crystal"""
    if not validate_element(element):
        sys.exit(1)
    if not validate_structure(structure):
        sys.exit(1)

    try:
        if lattice:
            atoms = bulk(element, structure, a=lattice)
        else:
            atoms = bulk(element, structure)
        print(f"Successfully built bulk crystal: {element} ({structure})")

        os.makedirs(output_dir, exist_ok=True)
        base_name = f"{element}_{structure}"
        output_file = os.path.join(output_dir, f"{base_name}.vasp")
        write(output_file, atoms, format="vasp")
        print(f"Saved: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error: Failed to build bulk crystal: {e}", file=sys.stderr)
        sys.exit(1)


def build_mol(mol_name: str, output_dir: str = ".") -> str:
    """Build molecule"""
    try:
        atoms = molecule(mol_name)
        print(f"Successfully built molecule: {mol_name}")

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{mol_name}.xyz")
        write(output_file, atoms)
        print(f"Saved: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error: Failed to build molecule '{mol_name}'. Please confirm molecule name is valid (e.g., H2O, CH3OH, C6H6)", file=sys.stderr)
        sys.exit(1)


def build_surface(element: str, structure: str, surface_type: str,
                   layers: int = 3, vacuum: float = 10.0,
                   lattice: float = None, output_dir: str = ".") -> str:
    """Build surface"""
    if not validate_element(element):
        sys.exit(1)
    if not validate_structure(structure):
        sys.exit(1)

    try:
        surface_indices = parse_surface_indices(surface_type)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if lattice:
            bulk_atoms = bulk(element, structure, a=lattice)
        else:
            bulk_atoms = bulk(element, structure)
        print(f"Successfully built bulk crystal: {element} ({structure})")

        atoms = surface(bulk_atoms, surface_indices, layers=layers, vacuum=vacuum)
        print(f"Successfully built surface: ({surface_indices}), {layers} layers, vacuum {vacuum}Å")

        os.makedirs(output_dir, exist_ok=True)
        # Use underscore format for surface indices for filename
        surface_name = ''.join(str(i) for i in surface_indices)
        base_name = f"{element}_{structure}_{surface_name}_surface"
        output_file = os.path.join(output_dir, f"{base_name}.vasp")
        write(output_file, atoms, format="vasp")
        print(f"Saved: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error: Failed to build surface: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Build crystal and surface structures using ASE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build silicon crystal
    python build_structure_ase.py --element Si --structure diamond --lattice 5.43

    # Build surface (both formats supported)
    python build_structure_ase.py --element Si --structure diamond --surface 111 --layers 3
    python build_structure_ase.py --element Si --structure diamond --surface 1,1,1 --layers 3

    # Build molecule
    python build_structure_ase.py --molecule H2O
        """
    )
    parser.add_argument("--element", help="Element symbol (e.g., Si, Fe)")
    parser.add_argument("--structure", help=f"Crystal structure (e.g., {', '.join(VALID_STRUCTURES)})")
    parser.add_argument("--lattice", type=float, help="Lattice constant (optional)")
    parser.add_argument("--molecule", help="Molecule name (e.g., H2O, CH3OH)")
    parser.add_argument("--surface", help="Surface indices (e.g., 111 or 1,1,1)")
    parser.add_argument("--layers", type=int, default=3, help="Number of atomic layers (default: 3)")
    parser.add_argument("--vacuum", type=float, default=10.0, help="Vacuum layer thickness (default: 10.0 Å)")
    parser.add_argument("--output", default=".", help="Output directory (default: current directory)")

    args = parser.parse_args()

    if args.molecule:
        build_mol(args.molecule, args.output)
    elif args.element and args.structure:
        if args.surface:
            build_surface(args.element, args.structure, args.surface,
                         args.layers, args.vacuum, args.lattice, args.output)
        else:
            build_bulk(args.element, args.structure, args.lattice, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
