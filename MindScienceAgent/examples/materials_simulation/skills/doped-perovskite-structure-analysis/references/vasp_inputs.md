# VASP Input Parameters Reference

Complete guide for VASP input file configuration for DFT structural optimization.

## INCAR Parameters

### Basic Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| SYSTEM | string | Job description |
| ISTART | 0 | Start from scratch |
| ICHARG | 2 | Charge from atomic densities |

### Electronic Optimization

| Parameter | Value | Description |
|-----------|-------|-------------|
| PREC | Accurate | Precision level |
| ENCUT | 520 | Plane-wave cutoff (eV) |
| EDIFF | 1E-6 | Electronic convergence |
| ALGO | Normal | Electronic minimization algorithm |
| LREAL | Auto | Real-space projection |

### Ionic Optimization

| Parameter | Value | Description |
|-----------|-------|-------------|
| EDIFFG | -0.01 | Ionic convergence (eV/Å) |
| NSW | 100 | Max ionic steps |
| IBRION | 2 | Conjugate gradient |
| ISIF | 3 | Relax cell + ions |

### Symmetry & Output

| Parameter | Value | Description |
|-----------|-------|-------------|
| ISYM | 0 | No symmetry |
| LORBIT | 11 | DOS output |
| LWAVE | .TRUE. | Write WAVECAR |
| LCHARG | .TRUE. | Write CHGCAR |

## KPOINTS File

### Gamma-centered Grid

```
Automatic mesh
0
Gamma
4 4 4
0 0 0
```

### Monkhorst-Pack Grid

```
Automatic mesh
0
Monkhorst-Pack
4 4 4
0 0 0
```

### K-point Density Guidelines

| System Size | K-mesh |
|-------------|--------|
| Primitive cell | 6x6x6 |
| 2x2x2 supercell | 4x4x4 |
| 3x3x3 supercell | 2x2x2 |

## POTCAR Generation

Using pymatgen:

```python
from pymatgen.io.vasp.sets import MPRelaxSet

# Standard relaxation set
vasp_input = MPRelaxSet(structure)
vasp_input.potcar.write_file("POTCAR")

# Custom potential selection
from pymatgen.io.vasp import Potcar
potcar = Potcar(["Sr_sv", "Ti_pv", "O", "Ba_sv"])
potcar.write_file("POTCAR")
```

### Recommended PAW Potentials

| Element | Potential | Notes |
|---------|-----------|-------|
| Sr | Sr_sv | Semi-core states |
| Ti | Ti_pv | Semi-core p states |
| O | O | Standard |
| Ba | Ba_sv | Semi-core states |

## Job Submission Scripts

### SLURM (sbatch)

```bash
#!/bin/bash
#SBATCH --job-name=vasp_relax
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=72:00:00
#SBATCH --partition=compute

module load vasp/6.3.0
mpirun vasp_std > vasp.out
```

### PBS (qsub)

```bash
#!/bin/bash
#PBS -N vasp_relax
#PBS -l nodes=2:ppn=24
#PBS -l walltime=72:00:00

module load vasp/6.3.0
mpirun vasp_std > vasp.out
```

## Convergence Checks

Check OUTCAR for:

1. **Electronic convergence**: Look for "EDIFF" criterion met
2. **Ionic convergence**: Look for "reached required accuracy"
3. **Total energy**: Should be stable in final steps
4. **Forces**: Should be < |EDIFFG| on all atoms

```bash
# Quick convergence check
grep "EDIFF" OUTCAR
grep "reached required accuracy" OUTCAR
grep "Total CPU time used" OUTCAR
```
