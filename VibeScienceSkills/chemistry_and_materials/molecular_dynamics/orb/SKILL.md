---
name: orb
description: orb is a machine learning force field (MLFF) based on graph neural networks (GNNs), designed as a universal interatomic potential model for various simulation tasks including geometry optimization, Monte Carlo simulations, and molecular dynamics simulations. Use this model when you need to predict total energy, atomic forces, and unit cell stress for inorganic materials and molecular systems.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team

---

# Orb

## Overview

Orb is a deep learning model for materials modeling that applies graph neural network architecture to learn the complexity of interatomic interactions. As a machine learning force field (MLFF), Orb takes as input a graph structure containing atomic positions, types, and system configuration (such as unit cell size and boundary conditions), and outputs the total energy of the system, force vectors for each atom, and unit cell stress.

Compared to existing open-source neural network potential models (such as MACE), Orb achieves a 3-6 times speed improvement at large system scales. In the Matbench Discovery benchmark, Orb reduced errors by 31% compared to other methods and became the state-of-the-art model. Orb performs excellently in zero-shot evaluation, remaining stable even in molecular dynamics simulations of high-temperature aperiodic molecules without fine-tuning for specific tasks.

This skill provides inference capabilities for the Orb model on Ascend NPU using the MindSpore framework.

---

## When to Use

- **Materials property prediction**: Predict energy, forces, and stress for crystalline materials and molecules
- **Molecular dynamics simulations**: Run stable MD simulations for inorganic systems
- **Geometry optimization**: Optimize atomic configurations to find minimum energy structures
- **Monte Carlo simulations**: Perform statistical sampling of material configurations
- **Zero-shot material discovery**: Apply pre-trained model to novel systems without fine-tuning

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | ASE trajectory SQLite databases (.db) or atomic structure files (POSCAR, .cif, .xyz) |
| Data Size   | Varies by system; pre-trained model works on diverse compositions |
| Data Source | Download from [dataset link](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/dataset/) |

#### Data Acquisition Methods

1. **Pre-trained model inference**: Download pre-trained checkpoint `orb-mptraj-only-v2.ckpt` from [model link](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/orb_ckpts/)
2. **Custom data**: Prepare atomic structure files in standard formats (POSCAR, CIF, XYZ)

#### Data Preprocessing

- Convert atomic structures to graph representation using the model's featurization utilities
- Ensure atomic positions, species, and unit cell information are properly formatted
- For ASE database format, use the bundled `ase_dataset.py` for loading

---

### 2. Environment Configuration and Dependencies

#### Environment Requirements

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Python Version | Python 3.8+ (tested with MindSpore 2.7.0)   |
| Framework    | MindSpore 2.7.0                               |
| Hardware     | Ascend NPU (910 series recommended)           |
| Memory       | At least 16GB RAM recommended                 |
| Disk Space   | At least 5GB for model checkpoints and outputs |

#### Dependency Installation

```bash
# Install MindSpore (follow official guide)
pip install mindspore==2.7.0

# Install MindScience (provides foundational components for equivariant computations)
# See: https://atomgit.com/mindspore-lab/mindscience

# Install other dependencies
pip install -r requirement.txt
```

#### Installation Steps

1. **Install MindSpore**: Follow the official installation guide at `https://www.mindspore.cn/install`
2. **Install MindScience**: See `https://atomgit.com/mindspore-lab/mindscience`
3. **Install Python dependencies**: `pip install -r requirement.txt`
4. **Download model checkpoint**: Place `orb-mptraj-only-v2.ckpt` under `orb_ckpts/` directory

#### Directory Structure

```
orb/
├── orb_ckpts/
│   └── orb-mptraj-only-v2.ckpt    # Pre-trained ORB checkpoint
├── configs/
│   └── config_eval.yaml           # Inference / evaluation configuration
├── src/                           # Core code for data processing
│   ├── ase_dataset.py             # Load and wrap ASE datasets
│   ├── atomic_system.py           # Data structures for atomic systems
│   ├── featurization_utilities.py # Convert atomic systems into model input
│   └── pretrained.py              # Interfaces for building and loading pre-trained models
├── models/                        # Model definitions
│   ├── gns.py                     # Graph Network Simulator structures
│   └── orb.py                     # Main ORB architecture (encoder + heads)
├── evaluate.py                    # Entry script for model inference / evaluation
└── requirement.txt                # Python dependency list
```

#### End-to-end checklist

| Step | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 1    | Install MindSpore 2.7.0 and MindScience per official guides |
| 2    | Create working directory and download pre-trained checkpoint |
| 3    | Prepare input structure files (POSCAR, CIF, or XYZ)         |
| 4    | Configure `configs/config_eval.yaml` with checkpoint path   |
| 5    | Run inference: `python evaluate.py`                         |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type         | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| Functional Limitations  | Requires atomic structure input; cannot generate structures from scratch |
| Performance Limitations | Inference time scales with number of atoms; larger systems take longer |
| Scale Limitations       | Recommended maximum ~1000 atoms per inference for typical hardware |
| Input Format            | Must be valid structure files with proper atom coordinates  |

#### Notes

- **Note 1**: Orb runs on MindSpore framework with Ascend backend. The device can be selected via the `device_target` field in config files.
- **Note 2**: For NPU inference, ensure Ascend NPU is properly configured and visible to MindSpore.
- **Note 3**: The pre-trained model (`orb-mptraj-only-v2.ckpt`) is trained on the mptraj dataset and works well for general inorganic materials.
- **Note 4**: Fine-tuning is available for domain-specific applications, but this skill focuses on inference with the pre-trained model.

---

### 4. Model Invocation Guide

#### Model Initialization

Load the pre-trained Orb model using the provided interface:

```python
import sys
sys.path.append('/path/to/orb/src')
sys.path.append('/path/to/orb/models')

from pretrained import OrbModel

# Initialize model with pre-trained checkpoint
model = OrbModel(
    checkpoint_path='/path/to/orb_ckpts/orb-mptraj-only-v2.ckpt',
    device_target='Ascend'  # or 'CPU'
)
```

#### Running Inference

Use the evaluate.py script for inference:

```bash
# Configure config_eval.yaml with your input and checkpoint paths
# Then run:
python evaluate.py --config configs/config_eval.yaml
```

Or use Python API directly:

```python
import numpy as np
from atomic_system import AtomicSystem
from featurization_utilities import atomic_system_to_graph

# Prepare input structure (example with ASE atoms)
from ase import Atoms
atoms = Atoms('Fe2O3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 1]], cell=[5,5,5])

# Convert to graph representation
graph = atomic_system_to_graph(atoms)

# Run inference
outputs = model.predict(graph)
energy = outputs['energy']  # Total energy
forces = outputs['forces']  # Atomic forces (N x 3)
stress = outputs['stress']  # Unit cell stress (6 components)
```

#### Result Post-processing

Output includes:
- **energy**: Total system energy (scalar)
- **forces**: Force vectors for each atom (N × 3 array)
- **stress**: Unit cell stress tensor (6 components: xx, yy, zz, xy, xz, yz)

---

## Reference Resources

- **MindSpore Official**: https://www.mindspore.cn/
- **MindScience**: https://atomgit.com/mindspore-lab/mindscience
- **Orb Model Repository**: https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/
- **Dataset Download**: https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/dataset/
- **Model Checkpoints**: https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/orb_ckpts/