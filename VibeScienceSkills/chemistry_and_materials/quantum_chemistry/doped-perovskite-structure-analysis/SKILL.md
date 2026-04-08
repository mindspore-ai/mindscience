---
name: doped-perovskite-structure-analysis
description: Step-by-step guide for obtaining the SrTiO3 crystal structure, constructing a Ba-doped supercell model, performing structural relaxation, and validating the structure with first-principles calculations to simulate XRD patterns. Use when the user needs to perform DFT calculations, build doping models, or analyze crystal structures using VASP and pymatgen.
---

# Doped Perovskite Structure Analysis Guide

A systematic workflow for studying dopant effects on crystal structures using first-principles calculations.

## Workflow Overview

```
Modeling -> Preliminary Optimization -> Precise Calculation -> Analysis
```

**Core Steps:**
1. **Acquisition**: Download crystal structure from Materials Project
2. **Model Construction**: Build supercell and create doping model with pymatgen
3. **Structural Optimization**: Perform DFT relaxation using VASP
4. **Result Analysis**: Calculate and compare XRD patterns

## Step 1: Download Base Structure from Materials Project

Use pymatgen to fetch the SrTiO3 structure (mp-5229):

```python
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure

API_KEY = "your_mp_api_key_here"
mp_id = "mp-5229"  # SrTiO3

with MPRester(API_KEY) as mpr:
    structure_sto = mpr.get_structure_by_material_id(mp_id)
    structure_sto.to(filename="STO_primitive.cif", fmt="cif")
```

> **Alternative**: Download manually from Materials Project website in CIF format.

## Step 2: Construct Supercell and Doping Model

Build a 2x2x2 supercell and substitute one Sr with Ba:

```python
from pymatgen.core import Structure

# Load and create supercell
structure_primitive = Structure.from_file("STO_primitive.cif")
structure_supercell = structure_primitive * [[2,0,0], [0,2,0], [0,0,2]]

# Find Sr sites and perform substitution
sr_indices = [i for i, site in enumerate(structure_supercell) if site.species_string == "Sr"]
structure_doped = structure_supercell.copy()
structure_doped[sr_indices[0]] = "Ba", structure_doped[sr_indices[0]].coords

structure_doped.to(filename="Ba_STO_2x2x2_doped.cif", fmt="cif")
```

## Step 3: DFT Structural Optimization with VASP

### Prepare Input Files

**POSCAR** - Convert doped structure:
```python
from pymatgen.io.vasp import Poscar
poscar = Poscar(structure_doped)
poscar.write_file("POSCAR")
```

**KPOINTS** - Gamma-centered mesh for 2x2x2 supercell:
```
Automatic mesh
0
Gamma
4 4 4
0 0 0
```

**INCAR** - Key parameters:
```
SYSTEM = Ba doped SrTiO3 relaxation
PREC = Accurate
ENCUT = 520
EDIFF = 1E-6
EDIFFG = -0.01
NSW = 100
IBRION = 2
ISIF = 3
ISYM = 0
LREAL = Auto
ALGO = Normal
```

**POTCAR** - Generate using pymatgen:
```python
from pymatgen.io.vasp.sets import MPRelaxSet
vasp_input = MPRelaxSet(structure_doped)
```

### Submit and Monitor

Submit VASP job via HPC queue system. Check OUTCAR for convergence:
- "reached required accuracy"
- "Total CPU time used"

## Step 4: XRD Pattern Analysis

Calculate and compare XRD patterns using pymatgen:

```python
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.vasp import Poscar
import matplotlib.pyplot as plt

# Read optimized structure
structure_optimized = Poscar.from_file("CONTCAR").structure

# Calculate XRD (Cu Ka radiation)
xrd_calc = XRDCalculator(wavelength="CuKa")
pattern = xrd_calc.get_pattern(structure_optimized, two_theta_range=(10, 90))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(pattern.x, pattern.y, 'b-', label='Ba-doped SrTiO3')
plt.xlabel('2θ (degrees)')
plt.ylabel('Intensity (a.u.)')
plt.legend()
plt.savefig('XRD_pattern.png', dpi=300)
```

### Analysis Points

- **Peak Shift**: Ba²⁺ (1.61 Å) > Sr²⁺ (1.44 Å) → lattice expansion → peaks shift to smaller angles
- **Peak Shape**: Splitting/broadening may indicate symmetry reduction (cubic → tetragonal)
- **Intensity**: Minor changes may occur due to structure modification

## Quick Reference

| Step | Command/Action |
|------|---------------|
| Get structure | `python scripts/get_structure.py` |
| Build model | `python scripts/build_doped_model.py` |
| Prepare VASP | `python scripts/prepare_vasp_inputs.py` |
| Run VASP | `sbatch scripts/job_submit.sh` |
| Analyze XRD | `python scripts/analyze_xrd.py` |

## Detailed References

- **VASP Input Parameters**: See [references/vasp_inputs.md](references/vasp_inputs.md) for complete INCAR settings
- **XRD Analysis**: See [references/xrd_analysis.md](references/xrd_analysis.md) for advanced analysis techniques
