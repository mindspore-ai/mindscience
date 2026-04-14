---
name: mattersim
description: mattersim is a deep learning model for materials simulation developed by Microsoft. It predicts material properties, simulates molecular dynamics, and accelerates materials discovery across the periodic table. Use this model when you need to perform property prediction, energy calculations, or force field inference for crystalline materials, molecules, and nanostructures.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# MatterSim

## Overview

MatterSim is a state-of-the-art deep learning model for materials simulation developed by Microsoft Research. It combines graph neural network architectures with large-scale pre-training on diverse material systems to enable accurate prediction of material properties, forces, and energies across the periodic table.

The model is designed to accelerate materials discovery by providing fast and accurate predictions for:
- Formation energies and material stability
- Atomic forces and stress tensors
- Elastic properties and mechanical behavior
- Electronic structure properties
- Molecular dynamics simulations

MatterSim supports a wide range of elements and can handle crystalline materials, molecules, and nanostructures with varying sizes and compositions.

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits their task requirements.

- **Scenario 1**: Materials property prediction - Suitable for predicting formation energies, elastic constants, and other thermodynamic properties of materials
- **Scenario 2**: Force field inference - Suitable for generating atomic forces for molecular dynamics simulations
- **Scenario 3**: Materials screening - Suitable for high-throughput screening of candidate materials for applications in energy, catalysis, and electronics
- **Scenario 4**: Structure optimization - Suitable for finding stable crystal structures and minimum energy configurations

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | ASE-compatible formats (POSCAR, XYZ, CIF) or MatterSim-specific JSON |
| Data Size | Single structures or directories containing multiple structure files |
| Data Source | Materials Project, AFLOW, OQMD, or custom structure files |

#### Data Acquisition Methods

1. **Materials Project Download** - Access via Materials Project API for crystalline materials with computed properties
2. **AFLOW Database** - Download from aflow.org for high-throughput DFT calculations
3. **OQMD** - Access the Open Quantum Materials Database for formation energy data
4. **Custom CIF/POSCAR Files** - Prepare structure files using crystallography software (VESTA, ASE)

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Ensure structure files contain complete atomic positions and cell parameters
- **Step 2**: Verify element types are supported by MatterSim (check periodic table coverage)
- **Step 3**: For multi-structure datasets, organize files in a clean directory hierarchy
- **Step 4**: Validate structure files using ASE or similar tools before inference

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

| Component | Version |
| --------- | ----------------------------- |
| HDK       | 25.3.RC1 |
| CANN      | 8.3.RC1 |
| Python    | 3.10 |
| torch     | 2.6.0 |
| torch-npu | 2.6.0 |

#### Clone repository and apply patch

```bash
git clone https://gitcode.com/AI4Science/MaterialsChemistry.git
cd MaterialsChemistry/Pytorch/mattersim
git clone https://github.com/microsoft/mattersim.git && cd mattersim
git checkout 5b1ee33b615faae41abd56581336827b2a1c49d3
git apply ../patch/mattersim.patch
cp -rf ../test/ ./
```

#### Conda environment and Python dependencies

```bash
conda create --name mattersim_env python=3.10
conda activate mattersim_env
cd /path/to/mattersim
pip install -r requirements.txt
```

**Ascend deployment constraints:**

- Install **torch** and **torch-npu** at the versions in the table above, using CANN-aligned packages per the MaterialsChemistry README and Huawei documentation.
- Install HDK and CANN on the host before Python dependency installation.

#### Download model parameters

```bash
# Download pre-trained model weights (check README for exact command)
python download_weights.py
```

This downloads the pre-trained model checkpoint files.

#### Environment Requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend NPU |
| Memory | 16GB+ RAM recommended for large structures |
| Disk Space | ~2-5GB for model parameters, ~500MB for code |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | Clone MaterialsChemistry and apply the patch flow above |
| 2 | Install HDK/CANN; create conda env; `pip install -r requirements.txt` |
| 3 | Download model parameters: `python download_weights.py` |
| 4 | Prepare input structure file (POSCAR, CIF, or XYZ) |
| 5 | Run inference: `python predict.py --input your_structure.cif --output results/` |

**Optional NPU availability check** (when Ascend applies):

```bash
python -c "import torch; print(torch.__version__); print(getattr(torch, 'npu', None) and torch.npu.is_available())"
```

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Functional Limitations | Limited to elements in training data; may not accurately predict properties for exotic compositions |
| Performance Limitations | Larger structures require more memory and compute time |
| Scale Limitations | Optimal performance for structures up to ~500 atoms; larger systems may require chunking |
| Input Format | Requires properly formatted structure files with complete atomic coordinates |

#### Notes

- **Note 1**: MatterSim supports multiple model variants for different property predictions (energy, forces, elasticity). Select the appropriate model for your task.
- **Note 2 (NPU)**: For Ascend inference, ensure **HDK/CANN** and **`torch` / `torch-npu` versions match** per the LifeScience README. Verify NPU visibility with the optional check above.
- **Note 3**: For best accuracy, ensure input structures are properly relaxed or represent realistic configurations.
- **Note 4**: Model performance may vary for materials outside the training distribution; validate with known systems first.

---

### 4. Model Invocation Guide

#### Model Initialization

| Item | Example / value |
| ---- | ---------------- |
| Default model | mattersim_v1 (general-purpose) |
| Force model | mattersim_forces (for MD simulations) |
| Checkpoint | ./checkpoints/mattersim.pt |

#### Running examples (recommended path)

**Shell:**

```bash
cd /path/to/mattersim

# Property prediction from CIF file
python predict.py \
    --input "./structures/example.cif" \
    --output "./results/" \
    --model mattersim_v1

# Force prediction for molecular dynamics
python predict.py \
    --input "./structures/molecule.xyz" \
    --output "./results/" \
    --model mattersim_forces \
    --compute_forces
```

**Python (optional):** Import and use the model programmatically - see `predict.py` and `examples/` for detailed API usage.

#### Result Post-processing

- Output files are written to the specified `--output` directory
- Results include predicted properties in JSON format and optionally visualized structures
- For energy predictions: output includes formation energy per atom and total energy
- For force predictions: output includes atomic forces in eV/Å

---

## Reference Resources

- **GitCode MaterialsChemistry (MatterSim)**: https://gitcode.com/AI4Science/MaterialsChemistry/tree/main/Pytorch/mattersim
- **Official README**: https://gitcode.com/AI4Science/MaterialsChemistry/blob/main/Pytorch/mattersim/README.md
- **Additional reference**: https://github.com/microsoft/mattersim/tree/main
- **Paper**: MatterSim: A Deep Learning Model for Materials Simulation (Microsoft Research)