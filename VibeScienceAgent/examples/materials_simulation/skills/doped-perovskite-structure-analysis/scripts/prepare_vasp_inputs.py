#!/usr/bin/env python
"""
Prepare VASP input files from a structure file.

Usage:
    python prepare_vasp_inputs.py --input Ba_STO.cif --encut 520 --kpoints 4x4x4
"""
import argparse
import os
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Kpoints, Incar


def prepare_vasp_inputs(
    input_file: str,
    encut: int = 520,
    kpoints: tuple = (4, 4, 4),
    system_name: str = "DFT calculation",
    output_dir: str = "."
):
    """
    Prepare VASP input files from a structure.
    
    Args:
        input_file: Input structure file (CIF or POSCAR)
        encut: Plane-wave cutoff energy (eV)
        kpoints: K-point mesh dimensions
        system_name: System name for INCAR
        output_dir: Output directory
    """
    # Load structure
    structure = Structure.from_file(input_file)
    print(f"Loaded structure: {structure.formula}")
    print(f"Number of atoms: {len(structure)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write POSCAR
    poscar = Poscar(structure)
    poscar_path = os.path.join(output_dir, "POSCAR")
    poscar.write_file(poscar_path)
    print(f"\nPOSCAR written to: {poscar_path}")
    
    # Write KPOINTS
    kpoints_obj = Kpoints.gamma_automatic(kpts=kpoints)
    kpoints_path = os.path.join(output_dir, "KPOINTS")
    kpoints_obj.write_file(kpoints_path)
    print(f"KPOINTS written to: {kpoints_path}")
    
    # Write INCAR
    incar_dict = {
        "SYSTEM": system_name,
        "ISTART": 0,
        "ICHARG": 2,
        "PREC": "Accurate",
        "ENCUT": encut,
        "EDIFF": 1e-6,
        "EDIFFG": -0.01,
        "NSW": 100,
        "IBRION": 2,
        "ISIF": 3,
        "ISYM": 0,
        "LREAL": "Auto",
        "ALGO": "Normal",
        "LORBIT": 11,
        "LWAVE": True,
        "LCHARG": True,
    }
    incar = Incar(incar_dict)
    incar_path = os.path.join(output_dir, "INCAR")
    incar.write_file(incar_path)
    print(f"INCAR written to: {incar_path}")
    
    # Print POTCAR generation hint
    print("\n" + "="*50)
    print("POTCAR generation:")
    print("="*50)
    print("Use pymatgen to generate POTCAR:")
    print("")
    print("from pymatgen.io.vasp.sets import MPRelaxSet")
    print(f"structure = Structure.from_file('{input_file}')")
    print("vasp_input = MPRelaxSet(structure)")
    print("vasp_input.potcar.write_file('POTCAR')")
    print("")
    print("Or manually merge POTCAR files:")
    elements = sorted(set([site.species_string for site in structure]))
    print(f"cat {' '.join([f'POTCAR_{e}' for e in elements])} > POTCAR")
    print("="*50)
    
    return structure


def main():
    parser = argparse.ArgumentParser(description="Prepare VASP input files")
    parser.add_argument("--input", "-i", required=True, help="Input structure file")
    parser.add_argument("--encut", type=int, default=520, help="Plane-wave cutoff (eV)")
    parser.add_argument("--kpoints", "-k", default="4x4x4", help="K-point mesh")
    parser.add_argument("--system", "-s", default="DFT calculation", help="System name")
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    # Parse kpoints
    kpoints = tuple(map(int, args.kpoints.split("x")))
    
    prepare_vasp_inputs(
        args.input, args.encut, kpoints, args.system, args.output
    )


if __name__ == "__main__":
    main()
