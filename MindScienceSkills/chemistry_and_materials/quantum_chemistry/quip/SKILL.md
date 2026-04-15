---
name: quip
description: quip (QUantum mechanics and Interatomic Potentials) is a software package for machine learning interatomic potentials. Use this model when you need to perform atomistic simulations, train Gaussian Approximation Potentials (GAP), or evaluate machine learning force fields for molecular dynamics.
license: GPL-3.0
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend
---

# QUIP

## Overview

QUIP (QUantum mechanics and Interatomic Potentials) is a software package developed by the libAtoms research group for machine learning interatomic potentials and atomistic simulations. It provides a comprehensive framework for:

- **Gaussian Approximation Potentials (GAP)**: A systematic approach to creating accurate machine learning potentials that can reproduce quantum mechanical (DFT) energies and forces with near-ab initio accuracy
- **Smooth Overlap of Atomic Positions (SOAP)**: A descriptor-based representation for atomistic structures that enables efficient comparison of local atomic environments
- **Molecular Dynamics Simulations**: Running atomistic simulations using machine learning-based force fields
- **Potential Energy Surface Fitting**: Training potentials from quantum mechanical reference data (DFT calculations)

QUIP bridges the gap between ab initio accuracy and classical force field efficiency, making it widely used in computational chemistry and materials science.

---

## When to Use

### Hardware Requirements

This model requires Ascend hardware. Before running, please verify that your device is Ascend:

```python
import subprocess

def check_npu_device():
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("Ascend not detected. This model requires Ascend hardware.")
    except FileNotFoundError:
        raise RuntimeError("npu-smi command not found. Please ensure Ascend driver is installed.")

check_npu_device()
```

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits your task requirements.

- **Scenario 1**: Machine learning potential training - Suitable for fitting interatomic potentials from DFT or ab initio reference data
- **Scenario 2**: Molecular dynamics simulations - Suitable for running atomistic simulations with ML-based force fields
- **Scenario 3**: Potential evaluation - Suitable for evaluating energies and forces using pre-trained GAP models
- **Scenario 4**: Materials modeling - Suitable for studying properties of materials at the atomic level

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | XYZ, Extended XYZ (.extxyz), or PDB format for atomic configurations |
| Reference Data | Training configurations with energies and forces from DFT calculations |
| Model Files | Pre-trained GAP model files (.xml) for inference |
| Data Size | Varies based on system size and number of configurations |
| Data Source | DFT calculations (VASP, Quantum ESPRESSO, CP2K), molecular dynamics trajectories, or experimental structures |

#### Data Acquisition Methods

1. **DFT Calculations** - Generate reference data from density functional theory (VASP, Quantum ESPRESSO, CP2K)
2. **Molecular Dynamics** - Sample configurations from classical or ab initio MD
3. **Pre-trained Models** - Download pre-trained GAP models from QUIP models repository or libAtoms

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Prepare atomic configurations in XYZ or extended XYZ format
- **Step 2**: Ensure configurations include energy and force information for training
- **Step 3**: For inference, prepare the system configuration and load pre-trained model
- **Step 4**: Verify atomic species and coordinate formats match model requirements

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

**Note**: QUIP is a **Fortran/C** package for interatomic potentials. The README lists **gfortran / ifort** and **meson** for building. It is **not** a PyTorch project; there is no `torch` / `torch-npu` stack.

| Component | Version |
| --------- | ------------------------------ |
| HDK       | 25.2.3 |
| CANN      | 8.3.RC1 |
| gfortran  | 4.4 |
| ifort     | 11.1 |
| Python    | 3.9 |
| torch     | N/A |
| torch-npu | N/A |

#### Clone repository

```bash
git clone https://gitcode.com/AI4Science/QUIP.git
cd QUIP
# Mirror listed in README: https://atomgit.com/gmq123/QUIP.git
```

#### Build (README summary)

```bash
conda create -n quip python=3.9
conda activate quip
# Install OS packages per README (e.g. gcc/g++/make, openblas-devel); pip install "f90wrap>=0.3.0" ase>=3.17.0
meson setup builddir -Dgap=true -Dmpi=false
meson compile -C builddir
```

**Note**: QUIP is a Fortran/C codebase. Dependencies include **gfortran or ifort**, **OpenBLAS**, **meson**, **f90wrap**, **ASE**, and related build tools (see GitCode README).

#### Environment Requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Host aligned with GitCode README (HDK/CANN guidance where applicable) |
| Memory | 8GB+ RAM recommended for training; 4GB+ for lighter inference |
| Disk Space | ~500MB for code, additional space for models and datasets |
| Compiler | Fortran compiler (gfortran recommended), build toolchain per README |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | `git clone https://gitcode.com/AI4Science/QUIP.git` |
| 2 | Install system dependencies per README (compilers, OpenBLAS, etc.) |
| 3 | `conda create -n quip python=3.9` and `conda activate quip` |
| 4 | `pip install "f90wrap>=0.3.0"`, `ase`, and other Python deps from README |
| 5 | `meson setup builddir -Dgap=true -Dmpi=false` and `meson compile -C builddir` |
| 6 | Run README test commands to verify the build |
| 7 | Prepare input configuration file |
| 8 | Run inference or training |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Training Data | Requires sufficient reference data (DFT calculations) for accurate potential fitting |
| Model Size | Pre-trained models can be large (hundreds of MB to GB) |
| System Size | Performance scales with number of atoms; very large systems may require specialized implementations |
| Domain Transfer | Potentials are typically trained for specific element combinations |
| Build Complexity | Requires compilation of Fortran/C components |

#### Additional Notes

- QUIP uses its own XML format for storing GAP models
- The `quip` command-line tool provides various utilities for potential evaluation
- Python API is available via the `quippy` module
- For production MD simulations, consider using QUIP with LAMMPS
- Integration with ASE (Atomic Simulation Environment) is supported

---

### 4. Inference Usage

This module provides specific instructions for running inference with QUIP.

#### Command-Line Usage

```bash
# Evaluate energies and forces using a pre-trained GAP model
quip atat.xyz potential_file=my_gap_model.xml

# Run molecular dynamics simulation with GAP potential
quip md init.xyz T=300 steps=10000 potential_file=gap_model.xml

# Optimize atomic structure
quip minimize init.xyz potential_file=gap_model.xml
```

#### Python API Example

```python
from quippy import Potential, Atoms

# Load pre-trained GAP model
pot = Potential("my_gap_model.xml")

# Load atomic configuration from file
atoms = Atoms("path/to/config.xyz")

# Set the calculator
atoms.set_calculator(pot)

# Get energy
energy = atoms.get_potential_energy()

# Get forces
forces = atoms.get_forces()

# Get stress tensor (optional)
stress = atoms.get_stress()

print(f"Energy: {energy} eV")
print(f"Forces shape: {forces.shape}")
print(f"Stress: {stress}")
```

#### Creating Configurations Programmatically

```python
from quippy import Atoms
import numpy as np

# Create a simple configuration (e.g., water molecule)
atoms = Atoms(natoms=3)
atoms.numbers = [8, 1, 1]  # O, H, H
atoms.positions = np.array([
    [0.0, 0.0, 0.0],
    [0.96, 0.0, 0.0],
    [-0.24, 0.93, 0.0]
])
atoms.cell = np.diag([10.0, 10.0, 10.0])
atoms.pbc = [True, True, True]
```

---

## Reference Resources

- **GitCode (primary)**: https://gitcode.com/AI4Science/QUIP
- **Official README**: https://gitcode.com/AI4Science/QUIP/blob/main/README.md
- **Additional reference**: https://github.com/libAtoms/QUIP/tree/main
- **QUIP Wiki**: https://github.com/libAtoms/QUIP/wiki
- **GAP Documentation**: See libAtoms documentation for Gaussian Approximation Potentials
- **Examples**: See the `examples/` directory in the repository

---

## License

QUIP is released under the **GPL-3.0** license. See the LICENSE file in the repository for details.

---

*This skill was created following the model-skill-creator template. For the current verified host stack, follow the GitCode README.*