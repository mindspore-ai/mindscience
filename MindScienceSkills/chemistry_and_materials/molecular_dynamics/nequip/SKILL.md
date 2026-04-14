---
name: nequip
description: nequip (Neural Equivariant Interatomic Potential) is a molecular potential prediction model based on E(3)-Equivariant Graph Neural Networks. Its core idea is to ensure physical consistency — that the model's predictions remain invariant under physical symmetries such as rotation, translation, and reflection. Use this model when you need to predict molecular energies and atomic forces for molecular dynamics simulations and materials discovery.
license: MIT
metadata:
    skill-author: MindSpore Science Team

---

# NequIP

## Overview

NequIP (Neural Equivariant Interatomic Potential) is a deep learning model for molecular potential energy surface prediction built upon **E(3)-Equivariant Graph Neural Networks**. The model ensures physical consistency by maintaining equivariance under 3D spatial transformations (rotation, translation, and reflection).

This skill provides inference capabilities for predicting molecular energies and atomic forces on Ascend NPU and GPU backends.

---

## When to Use

- **Molecular potential energy surface modeling**: Predict potential energies for molecular configurations
- **Atomic force prediction**: Calculate atomic forces for molecular dynamics simulations
- **Machine learning force field development**: Train and deploy ML-based force fields for molecular systems
- **Materials discovery**: Predict properties of molecular and material systems with quantum-chemical accuracy

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | NumPy compressed format (.npz) containing molecular structures |
| Data Size   | Varies based on molecular system; supports variable number of configurations |
| Data Source | RMD17 dataset (Revised MD-17) for training; custom NPZ files for inference |

#### Data Format Specification

The model expects input data in `.npz` format with the following fields:

| Field Name | Shape | Type | Description |
|-------------|--------|------|-------------|
| `nuclear_charges` | `(N_atoms,)` | int | Atomic numbers (Z) of each atom |
| `coords` | `(N_config, N_atoms, 3)` | float | Atomic coordinates (Å) |
| `energies` | `(N_config,)` | float | Scalar potential energy (kcal/mol) |
| `forces` | `(N_config, N_atoms, 3)` | float | Atomic forces (kcal/mol/Å) |

#### Data Preprocessing

- Ensure atomic coordinates are in Angstroms (Å)
- Provide atomic numbers as nuclear charges
- For inference, prepare NPZ files with `nuclear_charges` and `coords` fields

---

### 2. Environment Configuration and Dependencies

#### Verified Ascend stack (reference)

| Component | Version     |
| --------- | ----------- |
| HDK       | (per MindSpore installation) |
| CANN      | (per MindSpore installation) |
| Python    | 3.8+        |
| MindSpore | >= 2.7.0    |
| MindScience | (includes MindChem modules) |

Install **MindSpore** and **MindScience** per official documentation:
- MindSpore: https://www.mindspore.cn/install
- MindScience: https://atomgit.com/mindspore-lab/mindscience

#### Environment Requirements (hardware / disk)

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Hardware    | Ascend NPU (910 series) or GPU                |
| Memory      | At least 16GB RAM recommended                  |
| Disk Space  | At least 5GB for model checkpoints and outputs |

#### Installation Steps

1. **Install MindSpore**: Follow official installation guide at https://www.mindspore.cn/install
2. **Install MindScience**: Follow guide at https://atomgit.com/mindspore-lab/mindscience
3. **Clone the NequIP repository** (if using LifeScience version):
   ```bash
   git clone https://gitcode.com/AI4Science/LifeScience.git
   cd LifeScience/applications/nequip
   ```
4. **Prepare configuration**: Modify `rmd.yaml` for your dataset path and parameters

#### End-to-end checklist

| Step | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 1    | Install MindSpore >= 2.7.0 and MindScience per official guides |
| 2    | Clone LifeScience repository and navigate to `applications/nequip` |
| 3    | Prepare your NPZ dataset or use the bundled RMD17 uracil subset |
| 4    | Configure `rmd.yaml` with data path and model parameters    |
| 5    | Run inference: `python predict.py --config_file_path ./rmd.yaml --device_target Ascend --device_id 0` |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type         | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| Functional Limitations  | Requires atomic coordinates and nuclear charges as input; predicts energy and forces only |
| Performance Limitations | Inference time scales with number of atoms and configurations |
| Scale Limitations       | Recommended maximum ~1000 atoms per configuration for typical hardware |
| Input Format            | Must be valid NPZ format with proper coordinate arrays      |

#### Notes

- **Note 1**: NequIP uses E(3)-equivariant message passing to ensure physical consistency. The model is equivariant under rotation, translation, and reflection.
- **Note 2**: For NPU inference, ensure MindSpore and MindScience are properly installed with Ascend support.
- **Note 3**: The model achieves high data efficiency, requiring fewer training samples than traditional methods to reach quantum-chemical accuracy.
- **Note 4**: Energy is predicted in kcal/mol; forces are derived as the negative gradient of energy with respect to atomic positions.

---

### 4. Model Invocation Guide

#### Model Initialization

The model is initialized via the configuration file (`rmd.yaml`). Key parameters include:

| Parameter | Description |
| --------- | ------------|
| `data_file_path` | Path to NPZ dataset |
| `batch_size` | Training/inference batch size |
| `learning_rate` | Learning rate (training only) |
| `epochs` | Number of training epochs |
| `model` | Model architecture settings |

#### Running Inference

**From the NequIP directory:**

```bash
cd /path/to/LifeScience/applications/nequip
python predict.py --config_file_path ./rmd.yaml --mode GRAPH --device_target Ascend --device_id 0 --dtype float32
```

**Parameter description:**

- `--config_file_path`: Path to the configuration file (rmd.yaml)
- `--mode`: Execution mode, `GRAPH` for high-performance static graph
- `--device_target`: Target device type; `Ascend` or `GPU` for inference
- `--device_id`: Device index
- `--dtype`: Calculation precision type (e.g., `float32`)

#### Result Post-processing

Output includes:
- **Energy predictions**: Scalar potential energy for each configuration
- **Force predictions**: Atomic forces for each atom in each configuration

Results are saved according to the configuration in `rmd.yaml` and `predict.py`. Check the output directory specified in the configuration for prediction results.

---

## Reference Resources

- **NequIP Paper**: https://arxiv.org/abs/2101.03164
- **LifeScience Repository**: https://gitcode.com/AI4Science/LifeScience/tree/main/applications/nequip
- **MindSpore Installation**: https://www.mindspore.cn/install
- **MindScience**: https://atomgit.com/mindspore-lab/mindscience
- **RMD17 Dataset**: https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038