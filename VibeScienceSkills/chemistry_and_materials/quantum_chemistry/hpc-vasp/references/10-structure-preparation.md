# VASP Structure Preparation And Input Generation

## Contents

- Structure file acquisition
- Structure file conversion
- VASP input file generation
- qvasp toolkit reference

## Structure File Acquisition

Obtain initial crystal structure files before generating VASP inputs. Three primary paths:

### Path A: Download from Materials Project

Requires a valid Materials Project API key (32-character hex string from materialsproject.org dashboard).

```bash
# By material ID
python download_from_mp.py --api-key YOUR_API_KEY --material-id mp-149

# By formula (downloads first match)
python download_from_mp.py --api-key YOUR_API_KEY --formula SiO2 --output ./structures
```

Python API via pymatgen:
```python
from pymatgen.ext.matproj import MPRester
with MPRester(api_key) as mpr:
    structure = mpr.get_structure_by_material_id("mp-149")
    structure.to(filename="POSCAR", fmt="poscar")
```

### Path B: Build structures with ASE

Build bulk crystals, surfaces, and molecules directly.

```bash
# Bulk crystal
python build_structure_ase.py --element Si --structure diamond --lattice 5.43

# Surface (supports "111" or "1,1,1" format)
python build_structure_ase.py --element Si --structure diamond --surface 111 --layers 3

# Molecule
python build_structure_ase.py --molecule H2O
```

Supported ASE crystal structures: `diamond`, `fcc`, `bcc`, `hcp`, `sc`, `rocksalt`, `zincblende`, `wurtzite`

### Path C: Read local structure files

Convert from CIF, XYZ, VASP/POSCAR, XSD, PDB, and other formats.

```bash
# Using ASE
python read_with_ase.py --input my_structure.cif --output-dir ./converted

# Using pymatgen
python convert_structure.py --input my_structure.cif --formats cif poscar
```

Both tools output to VASP POSCAR format and validate minimum atomic distances.

## Structure File Conversion

When a structure file exists but needs format conversion:

| Input | Tool | Command |
|-------|------|---------|
| CIF → POSCAR | ASE | `ase.io.read('file.cif').write('POSCAR', format='vasp')` |
| CIF → POSCAR | pymatgen | `Structure.from_file('file.cif').to(filename='POSCAR', fmt='poscar')` |
| XYZ → POSCAR | ASE | `ase.io.read('file.xyz').write('POSCAR', format='vasp')` |
| Multiple formats | pymatgen | `Structure.from_file()` handles most formats |

Validation check: minimum atomic distance should exceed 0.5 Å. Values below indicate structural problems.

## VASP Input File Generation

### Four-file model overview

VASP requires four mutually-consistent input files:

| File | Concern |
|------|---------|
| `POSCAR` | lattice, species order, coordinates |
| `POTCAR` | pseudopotential datasets, must match POSCAR species order |
| `KPOINTS` | Brillouin-zone sampling |
| `INCAR` | run mode, electronic and ionic controls |

Species order must stay aligned between `POSCAR` and `POTCAR`. A mismatch causes physically wrong assignment even when parsing succeeds.

### Using qvasp (recommended)

qvasp is a VASP input generation toolkit. Install from https://qvasp.com/Installation.html

```bash
# POSCAR operations
qvasp -c2p              # CIF → POSCAR
qvasp -fix              # Fix atomic layers
qvasp -sc               # Create supercell

# POTCAR operations
qvasp -pbe Si           # Generate POTCAR with PBE functional
qvasp -pw91 Si          # Generate POTCAR with PW91 functional
qvasp -lda Si           # Generate POTCAR with LDA functional

# KPOINTS operations
qvasp -k density        # Automatic k-point mesh
qvasp -kline            # Line mode for band structure
qvasp -3k density       # 3D material k-mesh

# INCAR template generation
qvasp -relax            # Structure optimization
qvasp -scf              # Self-consistent calculation
qvasp -dos              # DOS calculation
qvasp -band             # Band structure
qvasp -hse              # HSE06 hybrid functional
qvasp -md               # Molecular dynamics
qvasp -elastic          # Elastic constants
qvasp -phonon           # Phonon spectrum

# Data extraction
qvasp -e                # Read energy from OUTCAR
qvasp -p2c              # POSCAR → CIF
qvasp -bandd            # Extract band data
qvasp -dosd             # Extract DOS data
qvasp -baderd           # Bader charge analysis
```

### ENCUT selection rule

Set `ENCUT` to 1.2–1.3× the maximum ENMAX value in the POTCAR files. Insufficient cutoff causes poor convergence and inconsistent total energies.

### Quick-start script

```bash
# Auto-generate structure optimization inputs
bash generate_vasp_inputs.sh --auto

# Interactive mode
bash generate_vasp_inputs.sh
```

This generates `INCAR` (relax), `KPOINTS` (mesh), and expects `POSCAR` and `POTCAR` to already exist. Review and modify `INCAR` for material-specific properties (magnetism, smearing, etc.).

## qvasp Toolkit Reference

For advanced VASP preprocessing and postprocessing:

| Command | Description |
|---------|-------------|
| `qvasp -vaspkit` | Call VASPKIT for advanced preprocessing |
| `qvasp -atomkit` | Call atomkit for structure manipulations |
| `qvasp -cls <POS>` | Cleave surface from POSCAR |
| `qvasp -mos <POS>` | Construct Moiré superlattice |
| `qvasp -out2arc` | Convert OUTCAR to trajectory arc file |
| `qvasp -orthcell` | Construct orthogonalized cell |
| `qvasp -redlat` | Redefine lattice vectors for PDOS |
| `qvasp -hej <POS1> <POS2>` | Construct heterojunction |
| `qvasp -nanotube <POS>` | Roll nanosheet to nanotube |
| `qvasp -3dband` | Obtain 3D band structure for 2D materials |

See: W. Yi, G. Tang et al. qvasp: A Flexible Toolkit for VASP Users in Materials Simulations. *Comput. Phys. Commun.*, 2020, 257, 107535.
