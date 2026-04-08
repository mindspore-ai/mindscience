---
name: chgnet
description: chgnet (Crystal Hamiltonian Graph neural Network) is a pretrained universal neural network potential for charge-informed atomistic modeling. Use this model when you need to predict energy, forces, stress, and magnetic moments for crystalline materials and molecules.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# CHGNet

## Overview

CHGNet (Crystal Hamiltonian Graph neural Network) is a pretrained universal neural network potential for **charge-informed atomistic modeling**. It predicts key atomic properties directly from crystal structure without requiring explicit charge calculations.

The model was pretrained on the GGA/GGA+U static and relaxation trajectories from the **Materials Project**, comprising more than **1.5 million structures** from **146k compounds** spanning the periodic table. CHGNet predicts:
- **Energy** (eV/atom)
- **Forces** (eV/Å)
- **Stress** (GPa)
- **Magnetic moments** (μB)

CHGNet uses a graph neural network architecture that represents crystal structures as graphs with atoms as nodes and interatomic bonds as edges, enabling efficient and accurate predictions for materials stability and properties.

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits your task requirements.

- **Scenario 1**: Materials property prediction - Suitable for predicting energy, forces, stress, and magnetic moments for crystalline materials
- **Scenario 2**: Structure optimization - Suitable for relaxing crystal structures to their energy-minimized configurations
- **Scenario 3**: Molecular dynamics simulation - Suitable for running charge-informed molecular dynamics at DFT-level accuracy
- **Scenario 4**: Materials discovery - Suitable for high-throughput screening and stability prediction (e.g., Matbench Discovery)

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | CIF (.cif), POSCAR, XYZ, or pymatgen Structure objects |
| Data Size | Single structures or batches for high-throughput prediction |
| Data Source | Materials Project, custom crystallographic data, or computational predictions |

#### Data Acquisition Methods

1. **Materials Project API** - Query structures from https://materialsproject.org/ using pymatgen
2. **CIF Files** - Download from crystallographic databases (ICSD, COD, AFLOW)
3. **Structure Files** - Use POSCAR (VASP), XSF (Quantum ESPRESSO), or other DFT output formats
4. **MPtrj Dataset** - Download pre-trained dataset from [figshare](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842)

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Ensure structure is converted to pymatgen Structure object using `Structure.from_file()` or `Structure.from_dict()`
- **Step 2**: Verify lattice parameters and atomic positions are in standard format
- **Step 3**: For magnetic materials, ensure initial magnetic moments are set if known
- **Step 4**: Place structure files in accessible directory path for inference

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

| Component | Version |
| --------- | ------------------------------ |
| HDK       | 25.2.0 |
| CANN      | 8.3.RC1 |
| Python    | 3.11 |
| torch     | 2.7.1 |
| torch-npu | 2.7.1 |

#### Clone repository and install (GitCode — primary)

```bash
git clone https://gitcode.com/AI4Science/CHGNet.git
cd CHGNet
# Mirror in README: https://atomgit.com/AI4Science/chgnet.git
conda create --name chgnet python=3.11
conda activate chgnet
pip install chgnet
pip install pyyaml torch_npu==2.7.1 decorator attrs psutil absl-py cloudpickle ml-dtypes scipy tornado
python npu_test.py
```

#### Runtime

Use **https://gitcode.com/AI4Science/CHGNet** README together with `npu_test.py` for the supported Ascend workflow.

---

### 3. Model Inference

This module explains how to load the pretrained model and perform inference on crystal structures.

#### Load Pretrained Model

```python
from chgnet.model.model import CHGNet

# Load default CHGNet (version 0.3.0, pretrained on MPtrj)
chgnet = CHGNet.load()

# Load specific model version
chgnet = CHGNet.load(model_name='r2scan')  # R2SCAN level model
```

#### Predict Structure Properties

```python
from chgnet.model.model import CHGNet
from pymatgen.core import Structure

chgnet = CHGNet.load()

# Load structure from file
structure = Structure.from_file('examples/mp-18767-LiMnO2.cif')

# Predict properties
prediction = chgnet.predict_structure(structure)

# Access results
print(f"Energy: {prediction['e']} eV/atom")
print(f"Forces: {prediction['f']} eV/Å")
print(f"Stress: {prediction['s']} GPa")
print(f"Magnetic moments: {prediction['m']} μB")
```

#### Batch Prediction

```python
from chgnet.model.model import CHGNet
from pymatgen.core import Structure
import glob

chgnet = CHGNet.load()

# Get all CIF files
cif_files = glob.glob('structures/*.cif')

# Batch predict
structures = [Structure.from_file(f) for f in cif_files]
predictions = chgnet.predict_structure(structures, task_id=0)
```

---

### 4. Structure Optimization

This module explains how to use CHGNet for structure relaxation/optimization.

```python
from chgnet.model import StructOptimizer
from pymatgen.core import Structure

# Initialize optimizer
relaxer = StructOptimizer()

# Load unrelaxed structure
structure = Structure.from_file('unrelaxed.cif')

# Run optimization
result = relaxer.relax(structure)

# Access results
final_structure = result["final_structure"]
trajectory = result["trajectory"]

print(f"Final energy: {trajectory.energies[-1]} eV")
print(f"Final structure:\n{final_structure}")
```

---

### 5. Molecular Dynamics

This module explains how to run molecular dynamics simulations using CHGNet.

```python
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import MolecularDynamics
from pymatgen.core import Structure
import warnings
warnings.filterwarnings("ignore")

# Load model and structure
chgnet = CHGNet.load()
structure = Structure.from_file("examples/mp-18767-LiMnO2.cif")

# Configure MD simulation
md = MolecularDynamics(
    atoms=structure,
    model=chgnet,
    ensemble="nvt",          # NVT or NPT
    temperature=1000,        # Kelvin
    timestep=2,              # femtoseconds
    trajectory="md_out.traj",
    logfile="md_out.log",
    loginterval=100,
)

# Run simulation (50 steps = 0.1 ps)
md.run(50)

# Device selection (if needed)
md_cpu = MolecularDynamics(
    atoms=structure,
    model=chgnet,
    use_device='npu',        # Device string per README (e.g. npu, cpu)
)
```

---

## Reference Resources

| Resource | Link |
|----------|------|
| GitCode (primary) | https://gitcode.com/AI4Science/CHGNet |
| Official README | https://gitcode.com/AI4Science/CHGNet/blob/main/README.md |
| Additional reference | https://github.com/CederGroupHub/chgnet |
| Official Documentation | https://chgnet.lbl.gov |
| API Documentation | https://cedergrouphub.github.io/chgnet/api |
| MPtrj Dataset | https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842 |
| LAMMPS Integration | https://github.com/advancesoftcorp/lammps/tree/based-on-lammps_2Aug2023/src/ML-CHGNET |

---

## End-to-end Checklist

- [ ] Install CHGNet: `pip install chgnet`
- [ ] Install dependencies: `pip install pymatgen ase`
- [ ] Prepare crystal structure (CIF, POSCAR, or pymatgen Structure)
- [ ] Load pretrained model: `chgnet = CHGNet.load()`
- [ ] Run prediction: `prediction = chgnet.predict_structure(structure)`
- [ ] Access results (energy, forces, stress, magmom)
- [ ] (Optional) Run structure optimization or MD simulation

---

## Model Limitations

- **Ascend workflow**: Follow the GitCode README and `npu_test.py` for the supported NPU path
- **Charge Prediction**: Magnetic moments are used as proxy for charge; explicit charge prediction not available
- **Force Accuracy**: MAE ~77 meV/Å; may not be sufficient for high-precision phonon calculations
- **System Size**: Performance scales with O(N) where N is number of atoms; very large systems may require chunking
- **Periodic Boundary Conditions**: Only supports periodic structures; molecular systems require padding