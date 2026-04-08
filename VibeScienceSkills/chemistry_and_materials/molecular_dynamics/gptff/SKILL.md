---
name: gptff
description: gptff (Graph-based Pretrained Transformer Force Field) is a deep learning model for simulating arbitrary inorganic materials with high precision and generalizability. Use this model when you need to perform fast energy, force, and stress calculations for inorganic crystal structures, or run structure optimization and molecular dynamics simulations.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# GPTFF

## Overview

GPTFF (Graph-based Pretrained Transformer Force Field) is a universal AI force field model designed for simulating arbitrary inorganic systems with high precision and generalizability. The model uses a graph-based transformer architecture to predict energy, forces, and stress tensors for crystalline materials, enabling applications in computational materials science, molecular dynamics, and structure optimization.

The model works as an ASE (Atomic Simulation Environment) calculator, making it compatible with a wide range of molecular dynamics and optimization algorithms. GPTFF supports various inorganic material systems and provides out-of-the-box predictions without requiring task-specific fine-tuning.

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits their task requirements.

- **Scenario 1**: Fast energy, force, and stress calculation - Suitable for rapid property prediction of inorganic crystal structures (energy in eV, forces in eV/Å, stress in GPa)
- **Scenario 2**: Structure optimization - Suitable for optimizing atomic positions and/or lattice vectors to find minimum energy configurations
- **Scenario 3**: Molecular dynamics simulations - Suitable for running NVT ensemble molecular dynamics using ASE integrators
- **Scenario 4**: High-throughput materials screening - Suitable for evaluating large numbers of structures in computational materials discovery workflows

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | POSCAR, CIF, or other formats supported by pymatgen (Structure.from_file) |
| Data Size | Single structure files or directories containing multiple structure files |
| Data Source | Materials Project, AFLOW, OQMD, custom computational results, or experimental structures |

#### Data Acquisition Methods

1. **Materials Project Download** - Access via https://materialsproject.org/ - Search for inorganic compounds, download CIF/POSCAR
2. **AFLOW Database** - Access via http://aflow.org/ - Large repository of inorganic crystal structures
3. **OQMD (Open Quantum Materials Database)** - Access via http://oqmd.org/ - DFT-computed structures
4. **Custom Structure Files** - Create POSCAR files using VASP, pymatgen, or other materials modeling tools

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Ensure structure file is in a format readable by pymatgen (POSCAR, CIF, vasprun.xml, etc.)
- **Step 2**: Verify the structure contains valid lattice vectors and atomic positions
- **Step 3**: Place structure file in an accessible directory path for inference
- **Step 4**: For ASE integration, use AseAtomsAdaptor to convert pymatgen Structure to ASE Atoms object

---

### 2. Environment Configuration and Dependencies

#### Component versions

| Component | Version |
| --------- | ------------------------------ |
| HDK       | 24.1.RC3 |
| CANN      | 8.2.RC1 |
| Python    | 3.11 |
| torch     | 2.6.0 |
| torch-npu | 2.6.0 |

#### Clone repository

```bash
git clone https://atomgit.com/AI4Science/gptff.git
cd gptff
```

#### Environment setup

1. **Create and activate conda environment**

```bash
conda create -n gptff python=3.11
conda activate gptff
```

Point dynamic loading at the env’s `lib` (adjust if your conda root is not `/root/miniconda3`):

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/gptff/lib:$LD_LIBRARY_PATH
```

2. **Install dependencies**

Install **torch** and **torch-npu** builds that match **CANN 8.2.RC1** (e.g. `torch==2.6.0` and the matching `torch-npu` wheel from Huawei documentation), then:

```bash
pip install -e .
pip install torch_npu pyyaml decorator attrs psutil absl-py cloudpickle ml-dtypes scipy tornado ase==3.21.1
pip install cython
python setup.py build_ext --inplace
```

#### Pipeline tuning (optional)

```bash
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
```

#### Run bundled entry script

The repository provides a top-level driver (often used for smoke tests or demos):

```bash
python model_inference.py
```

#### Environment requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend NPU |
| Memory | 8GB+ host RAM recommended; scale with structure size and device memory |
| Disk space | ~500MB+ for weights and dependencies |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | `git clone https://atomgit.com/AI4Science/gptff.git` |
| 2 | `conda create -n gptff python=3.11` / `conda activate gptff`; set `LD_LIBRARY_PATH` to env `lib` |
| 3 | Install **torch** / **torch-npu** for CANN; `pip install -e .`; extra pip deps; **Cython** + `python setup.py build_ext --inplace` |
| 4 | Optional: `CPU_AFFINITY_CONF`, `TASK_QUEUE_ENABLE` |
| 5 | Run `python model_inference.py` or the Python API below |
| 6 | Prepare structures (POSCAR, CIF, etc.) for custom runs |

#### Model weights download

Pretrained model weights are included in the `pretrained/` directory:
- `pretrained/gptff_v1.pth` - Version 1 model weights
- `pretrained/gptff_v2.pth` - Version 2 model weights (if available)

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Functional Limitations | Does not support molecular dynamics with LAMMPS (coming soon); only ASE-based MD |
| Performance Limitations | Large systems benefit from ample device memory and tuned batch settings |
| Scale Limitations | Performance scales with system size; very large systems may require memory optimization |
| Input Format | Only supports inorganic materials; not designed for organic molecules |

#### Notes

- **Note 1**: The model outputs energy in eV, forces in eV/Å, and stress in GPa (or kBar for training data)
- **Note 2**: For structure optimization with variable lattice vectors, use `ExpCellFilter`; for fixed lattice, use `BFGS` or `FIRE`
- **Note 3**: The model uses ASE calculator interface, making it compatible with all ASE optimizers and MD integrators
- **Note 4**: Reference energies (atom_refs) are pre-trained; custom atom_refs can be provided for domain-specific applications

---

### 4. Model Invocation Guide

#### Model Initialization

| Item | Example / value |
| ---- | ---------------- |
| Model weight path | `pretrained/gptff_v1.pth` or `pretrained/gptff_v2.pth` |
| Device | `'npu'` (Ascend) |

#### Running examples (recommended path)

**Python: Energy, Force, and Stress Calculation**

```python
from gptff.model.mpredict import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# Initialize the model and load weights
model_weight = "pretrained/gptff_v1.pth"
device = 'npu'
p = ASECalculator(model_weight, device)

# Load structure
struc = Structure.from_file('POSCAR_structure')

# Convert to ASE atoms
adp = AseAtomsAdaptor()
atoms = adp.get_atoms(struc)
atoms.set_calculator(p)

# Get properties
energy = atoms.get_potential_energy()  # unit: eV
forces = atoms.get_forces()            # unit: eV/Å
stress = atoms.get_stress()            # unit: GPa
```

**Python: Structure Optimization (with lattice relaxation)**

```python
from gptff.model.mpredict import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter

model_weight = "pretrained/gptff_v1.pth"
device = 'npu'
p = ASECalculator(model_weight, device)

struc = Structure.from_file('POSCAR_structure')
adp = AseAtomsAdaptor()
atoms = adp.get_atoms(struc)
atoms.set_calculator(p)

# Optimize with variable cell
optimizer = ExpCellFilter(atoms)
FIRE(optimizer).run(fmax=0.01, steps=100)
```

**Python: Structure Optimization (fixed lattice)**

```python
from gptff.model.mpredict import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize.bfgs import BFGS

model_weight = "pretrained/gptff_v1.pth"
device = 'npu'
p = ASECalculator(model_weight, device)

struc = Structure.from_file('POSCAR_structure')
adp = AseAtomsAdaptor()
atoms = adp.get_atoms(struc)
atoms.set_calculator(p)

# Optimize only atomic positions
optimizer = BFGS(atoms)
optimizer.run(fmax=0.01, steps=1000)
```

**Python: Molecular Dynamics (NVT ensemble)**

```python
from gptff.model.mpredict import ASECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import units
from ase.md.nvtberendsen import NVTBerendsen
import os

model_weight = "pretrained/gptff_v1.pth"
device = 'npu'
p = ASECalculator(model_weight, device)

struc = Structure.from_file('POSCAR_structure')
adp = AseAtomsAdaptor()
atoms = adp.get_atoms(struc)
atoms.set_calculator(p)

# Setup output directory
save_dir = './results_path'
os.makedirs(save_dir, exist_ok=True)

# Run NVT dynamics
temp = 430  # temperature in K
dyn = NVTBerendsen(
    atoms=atoms,
    timestep=2 * units.fs,
    temperature=temp,
    taut=200 * units.fs,
    loginterval=20,
    logfile=os.path.join(save_dir, 'output.txt'),
    trajectory=os.path.join(save_dir, f'md_trajectory.trj'),
    append_trajectory=True
)
dyn.run(100000)
```

#### Result Post-processing

- **Energy**: Output in eV (electronvolts)
- **Forces**: Output in eV/Å (force per angstrom)
- **Stress**: Output in GPa (gigapascals); training data uses kBar
- **Trajectory files**: Saved in ASE trajectory format (.trj) for visualization with ASE or OVITO

---

## Reference resources

- **AtomGit (clone URL)**: https://atomgit.com/AI4Science/gptff
- **GitCode mirror**: https://gitcode.com/AI4Science/GPTFF
- **Upstream / lab**: https://github.com/atomly-materials-research-lab/GPTFF
- **Paper**: Xie et al., "GPTFF: A high-accuracy out-of-the-box universal AI force field for arbitrary inorganic materials", Science Bulletin (2024)
- **ASE Documentation**: `https://wiki.fysik.dtu.dk/ase/`
- **pymatgen Documentation**: `https://pymatgen.org/`