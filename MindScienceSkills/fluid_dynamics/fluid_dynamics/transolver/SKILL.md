---
name: transolver
description: transolver is a Transformer-based neural operator model for solving partial differential equations (PDEs) on general geometries. It uses self-attention mechanisms to model long-range interactions between spatial points, making it suitable for irregular meshes, unstructured grids, and complex boundary conditions. Use this model when you need to perform fast PDE solving for physics-based simulations on Ascend NPU.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team

---

# Transolver

## Overview

Transolver is a deep learning model that solves partial differential equations (PDEs) on general geometries using a Transformer architecture. Unlike Fourier-based neural operators (e.g., FNO) that require regular periodic grids, Transolver can handle irregular meshes, unstructured grids, and complex boundary conditions common in engineering applications.

The model treats discrete mesh points or elements as tokens and uses self-attention to learn global mappings between physical fields. This makes it particularly suitable for structural mechanics, material deformation, and other engineering problems where the domain geometry is non-regular.

This skill provides inference capabilities for Transolver on Ascend NPU using the MindSpore framework, with a focus on hyper-elastic material (Elasticity) problem solving.

---

## When to Use

- **Structural mechanics analysis**: Solve elasticity problems for engineering structures with complex geometries
- **Material deformation modeling**: Predict displacement fields for hyper-elastic materials under load
- **Non-regular domain PDE solving**: Handle irregular meshes, finite element grids, or unstructured discretizations
- **Fast physics inference**: Replace expensive numerical solvers with data-driven predictions for repeated queries
- **Multi-physics coupling (optional)**: Extend to coupled PDE systems in fluid-structure interaction

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | Mesh-based data with node coordinates and field values (see Elasticity dataset structure) |
| Data Size   | Training: hundreds to thousands of samples; Testing: tens to hundreds of samples |
| Data Source | [Elasticity Dataset](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) from Google Drive |

#### Data Acquisition Methods

1. **Download from Google Drive**: Access the Elasticity dataset from the provided link
2. **MindScience Repository**: Clone MindFlow repository which contains sample data and training scripts

#### Data Preprocessing

- Save dataset to `./data/elasticity/Meshes/` directory
- Ensure mesh data includes node positions and boundary conditions
- Format should be compatible with MindFlow data loaders

---

### 2. Environment Configuration and Dependencies

#### Verified Ascend Stack (MindSpore)

| Component    | Version  |
| ------------ | -------- |
| Hardware     | Ascend NPU |
| MindSpore    | >= 2.7.1 |
| Python       | 3.8+ (recommended 3.10) |

#### Environment Setup

1. **Install MindSpore**: Ensure MindSpore >= 2.7.1 is installed on your Ascend NPU environment

```bash
pip install mindspore>=2.7.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. **Clone MindScience Repository**: Get the Transolver implementation from MindFlow

```bash
# Clone MindScience repository
git clone https://atomgit.com/mindspore-lab/mindscience.git
# Or directly access Transolver code
# https://atomgit.com/mindspore-lab/mindscience/tree/master/MindFlow/applications/data_driven/transolver
```

3. **Prepare Dataset**: Download Elasticity dataset and place in `./data/elasticity/Meshes/`

#### Environment Requirements (hardware / disk)

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Hardware    | Ascend NPU (910 series recommended)            |
| Memory      | At least 16GB RAM recommended                  |
| Disk Space  | At least 10GB for model checkpoints and data   |

#### End-to-end checklist

| Step | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 1    | Install MindSpore >= 2.7.1 on Ascend NPU environment        |
| 2    | Clone MindScience/MindFlow repository or obtain Transolver code |
| 3    | Download Elasticity dataset from Google Drive               |
| 4    | Place data in `./data/elasticity/Meshes/` directory         |
| 5    | Run inference: `python exp_elas.py --eval 1 --save_name elas_transolver` |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type         | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| Functional Limitations  | Requires mesh-based input data; designed for 2D Elasticity problems |
| Performance Limitations | Inference time scales with mesh resolution and complexity   |
| Scale Limitations       | Tested on moderate mesh sizes; very large meshes may require memory optimization |
| Framework               | MindSpore only (not PyTorch)                                |

#### Notes

- **Note 1**: Transolver is based on MindSpore framework, not PyTorch. Ensure proper MindSpore installation for Ascend NPU.
- **Note 2**: The model is designed for PDE solving on irregular geometries - it does not require periodic boundary conditions like FNO.
- **Note 3**: For training, use `python exp_elas.py --save_name elas_transolver`. For inference/evaluation, use `python exp_elas.py --eval 1 --save_name elas_transolver`.
- **Note 4**: Model checkpoints are saved to `./checkpoints/` directory.

---

### 4. Model Invocation Guide

#### Model Initialization

The Transolver model is initialized through the training/inference script. Model weights are loaded from checkpoint files.

#### Running Inference

From the Transolver directory (where `exp_elas.py` is located):

```bash
# Run evaluation/inference
python exp_elas.py --eval 1 --save_name elas_transolver
```

This will:
- Load the trained model from `./checkpoints/elas_transolver.ckpt`
- Run inference on the test dataset
- Output relative error metrics

#### Expected Output

```
100%|██████████| 200/200 [00:02<00:00, 67.58it/s]
rel_err : 0.00951484518358484
```

The `rel_err` value represents the relative error between predicted and ground truth displacement fields.

---

## Reference Resources

- **Transolver Paper**: [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/abs/2402.02366)
- **MindFlow Repository**: https://atomgit.com/mindspore-lab/mindscience/tree/master/MindFlow/applications/data_driven/transolver
- **MindSpore Official Website**: https://www.mindspore.cn/
- **Elasticity Dataset**: https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8