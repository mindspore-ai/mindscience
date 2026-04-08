---
name: schnet
description: schnet (Sharp Continuum Neural Network) is a deep learning model for quantum chemistry predictions and atomistic machine learning. Use this model when you need to predict molecular properties such as energy, forces, dipole moments, and polarizabilities for molecules and materials.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# SchNet

## Overview

SchNet (Sharp Continuum Neural Network) is a deep learning model for quantum chemistry predictions and atomistic machine learning. It uses continuous-filter convolutional neural networks to predict molecular properties such as energies, forces, dipole moments, and polarizabilities for molecules and materials. The model is part of the SchNetPack ecosystem and supports various property prediction tasks including molecular dynamics simulations.

SchNet represents atoms as continuous-filter feature vectors that are updated through interaction blocks, enabling accurate predictions of molecular properties while maintaining rotational equivariance. The model has been widely used in computational chemistry and materials science for property prediction, molecular dynamics, and drug discovery applications.

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits your task requirements.

- **Scenario 1**: Quantum chemistry property prediction - Suitable for predicting molecular properties such as energy, forces, dipole moments, and polarizabilities
- **Scenario 2**: Molecular dynamics simulation - Suitable for running force field predictions and molecular dynamics simulations
- **Scenario 3**: Materials property prediction - Suitable for predicting properties of crystalline materials and nanostructures

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | XYZ, PDB, or ASE-compatible molecular structure formats |
| Data Size | Varies by dataset (QM9: ~134k molecules, MD17: molecular dynamics trajectories) |
| Data Source | Benchmark datasets (QM9, MD17, ANI-1) or custom molecular structures |

#### Data Acquisition Methods

1. **Benchmark Datasets** - Use SchNetPack CLI to download built-in datasets (QM9, MD17, ANI-1)
2. **RCSB PDB** - Download molecular structures from https://www.rcsb.org/
3. **Custom XYZ Files** - Prepare molecular structures in XYZ format using molecular modeling software

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Ensure molecular structure is in a format compatible with ASE (Atomic Simulation Environment)
- **Step 2**: Verify all atomic positions and elements are correctly specified
- **Step 3**: For custom datasets, organize into SchNetPack-compatible format or use ASE to convert
- **Step 4**: Place data files in accessible directory path for inference

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

| Component | Version |
| --------- | ------------------------------ |
| HDK       | 25.2.0 |
| CANN      | 8.1.rc1 |
| Python    | 3.10.17 |
| torch     | 2.1.0 |
| torch-npu | 2.1.0.post17 |
| scikit-learn | 1.1.3 |

#### Clone repository and prepare code (Ascend)

```bash
git clone https://gitcode.com/AI4Science/SchNet.git
cd SchNet
# Per README: AscendHub image, torch/torch_npu wheels, PYG, CustomOp install.sh, CppExtensionInvocation, etc.
```

#### Conda environment and Python dependencies

```bash
# Create conda environment
conda create --name schnet_env python=3.10
conda activate schnet_env

# Install system dependencies
yum install -y gcc gcc-c++ libstdc++-devel libstdc++ make cmake openblas-devel openblas-static util-linux

# Install PyTorch and torch-npu (from GitCode README)
wget https://download.pytorch.org/whl/cpu/torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
wget https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.1.0/torch_npu-2.1.0.post17-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch_npu-2.1.0.post17-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# Install PYG components
pip3 install torch_cluster torch_geometric --no-build-isolation
pip3 install ase

# Install SchNet custom operators
export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:$CPLUS_INCLUDE_PATH
bash install.sh -v Ascend910B3
cd CustomOp/build_out
./custom_opp_openEuler_aarch64.run
cd ../../CppExtensionInvocation/
bash build_and_run.sh
```

#### Environment Requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend 910B3 (NPU) per GitCode README |
| Memory | Recommended 16GB+ RAM |
| Disk Space | ~5GB for models and datasets |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | Clone `https://gitcode.com/AI4Science/SchNet.git` |
| 2 | Install system dependencies, PyTorch, and torch-npu per README |
| 3 | Build and install custom NPU operators per README |
| 4 | Use built-in datasets or prepare custom molecular structures |
| 5 | Run inference: `python3 QM9.py` (per README) |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Functional Limitations | Requires atomic structure input; cannot predict from SMILES alone without structure conversion |
| Performance Limitations | Custom operators required for optimal NPU performance |
| Scale Limitations | Memory usage scales with number of atoms and batch size |
| Input Format | Requires ASE-compatible molecular structure formats |

#### Notes

- **Note 1**: For Ascend NPU, the custom radius operator must be built and installed for graph construction
- **Note 2**: For Ascend inference, HDK/CANN and torch/torch-npu versions must match the verified stack table; confirm device visibility with a short `torch.npu` probe if needed
- **Note 3**: Other SchnetPack distributions may publish different Python or PyTorch requirements; this skill follows the GitCode SchNet README

---

### 4. Model Invocation Guide

#### Model Initialization

| Item | Example / value |
| ---- | ---------------- |
| Entry Script (GitCode) | `python3 QM9.py` |

#### Running examples (recommended path)

**Shell (Ascend path):**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /path/to/SchNet
python3 QM9.py
```

#### Result Post-processing

- Results are written to the specified output directory
- For property prediction, outputs include predicted energy, forces, dipole moments, etc.
- Use ASE for visualization and further analysis of molecular structures

---

## Reference Resources

- **GitCode (primary)**: https://gitcode.com/AI4Science/SchNet
- **Official README**: https://gitcode.com/AI4Science/SchNet/blob/main/README.md
- **Additional reference**: https://github.com/atomistic-machine-learning/schnetpack
- **SchNet Paper**: Schütt et al., "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions"
- **ASE (Atomic Simulation Environment)**: `https://wiki.fysik.dtu.dk/ase/`