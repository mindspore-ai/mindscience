---
name: fno3d
description: fno3d (Fourier Neural Operator for 3D) is a deep learning model that solves the 3-dimensional Navier-Stokes equation using neural operator learning. It learns mappings between infinite-dimensional function spaces to solve partial differential equations, providing fast inference for computational fluid dynamics tasks. Use this model when you need to predict solutions to 3D PDEs efficiently, particularly for fluid flow simulation.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team

---

# FNO3D

## Overview

FNO3D (Fourier Neural Operator for 3D) is a deep learning model that solves the 3-dimensional Navier-Stokes equation using the Fourier Neural Operator architecture. Unlike traditional neural networks that learn mappings between finite-dimensional spaces, FNO can learn mappings between infinite-dimensional function spaces, making it particularly effective for solving partial differential equations (PDEs).

The model consists of three main components:
- **Lifting Layer**: Lifts the input to a higher-dimensional representation
- **Fourier Layers**: Apply Fourier transform, perform linear transformations on lower Fourier modes, filter out higher modes, then apply inverse Fourier transform
- **Decoding Layer**: Decodes the final output from the high-dimensional representation

This skill provides inference capabilities for running FNO3D on Huawei Ascend NPUs using MindSpore.

---

## When to Use

- **3D PDE solving**: Solve 3-dimensional partial differential equations, particularly the Navier-Stokes equation
- **Computational fluid dynamics**: Fast prediction of 3D fluid flow behavior without traditional numerical methods
- **Scientific computing**: Replace expensive finite element or finite difference methods with fast neural operator inference
- **Parametric PDE problems**: Learn operators that map initial conditions to solutions for a class of PDEs

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | NumPy arrays (.npy) containing initial vorticity and solution fields |
| Data Size   | Training: multiple samples with resolution 3D; Inference: single or batch of initial conditions |
| Data Source | [3D Navier-Stokes Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes_3d/) |

#### Data Acquisition Methods

1. **Official MindSpore Dataset**: Download from [data_driven/navier_stokes_3d/](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes_3d/)
2. **Custom Data**: Generate initial vorticity w0(x) for the 3D Navier-Stokes equation and prepare corresponding solution fields

#### Data Preprocessing

- Normalize input initial conditions to appropriate range
- Ensure input resolution matches model configuration
- Prepare data in shape [batch_size, 1, depth, height, width] for model input

---

### 2. Environment Configuration and Dependencies

#### Verified Ascend stack (reference)

| Component | Version     |
| --------- | ----------- |
| HDK       | [Per MindSpore requirements] |
| CANN      | [Per MindSpore requirements] |
| Python    | 3.8+        |
| MindSpore | >=2.7.0     |
| MindScience | 0.8.0     |

#### Environment Setup

```bash
# Install MindSpore
pip install mindspore==2.7.0

# Install MindScience (MindFlow)
pip install mindscience==0.8.0

# Clone MindScience repository for source code
git clone https://atomgit.com/mindspore-lab/mindscience.git
cd mindscience/MindFlow
```

#### Environment Requirements (hardware / disk)

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Python Version | 3.8+                                           |
| Hardware    | Ascend NPU with >32G memory                    |
| Memory      | At least 32GB RAM                              |
| Disk Space  | ~1GB for model and dataset                     |

#### Installation Steps

1. **Install MindSpore**: Follow official MindSpore installation guide for Ascend
2. **Install MindScience (MindFlow)**: Install mindscience==0.8.0 or clone and install from source
3. **Download dataset**: Get the 3D Navier-Stokes dataset from the official source
4. **Prepare config**: Configure model parameters in config file (default: `./configs/fno3d.yaml`)

---

### 3. Performance

| Parameter               | Ascend               |
|:----------------------:|:--------------------------:|
| Hardware                | Ascend 32G           |
| MindSpore version       | 2.7.0               |
| dataset                 | [3D Navier-Stokes Equation Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes_3d/)      |
| Parameters              | 6.5e6                  |
| Train Config            | batch_size=10, steps_per_epoch=100, epochs=150 |
| Evaluation Config       | batch_size=1      |
| Optimizer               | Adam                 |
| Train Loss(MSE)         | 0.01                |
| Evaluation Error(RMSE)  | 0.02                |
| Speed(ms/epoch)          | 12179                   |

---

### 4. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
|----------------|--------------|
| Functional Limitations | Limited to solving Navier-Stokes equation; other PDEs require model retraining |
| Performance Limitations | Inference speed depends on input resolution and batch size |
| Scale Limitations | Input resolution and time steps are limited by available GPU/NPU memory |

#### Notes

- **Note 1**: The model learns the operator mapping from w_t to w_(t+1), requiring iterative inference for multi-step predictions
- **Note 2**: Dataset must be saved in `./dataset` directory for default configuration to work
- **Note 3**: Configuration file path can be customized via `--config_file_path` parameter

---

### 5. Model Invocation Guide

#### Inference Code

```python
import mindspore as ms
from mindflow import load_yaml
from mindflow.common import get_ms_context

# Initialize MindSpore context
get_ms_context(mode=0, device_target="Ascend", device_id=0)

# Load configuration
config = load_yaml("./configs/fno3d.yaml")

# Model inference (example)
# Note: Load trained checkpoint and run inference on test data
# See train.py for complete training/inference workflow
```

#### Running Inference

```bash
export PYTHONPATH=$(cd ../../../../../ && pwd):$PYTHONPATH
python train.py --config_file_path ./configs/fno3d.yaml --mode GRAPH --device_target Ascend --device_id 0
```

Where:
- `--config_file_path`: Path to the parameter file (default: './configs/fno3d.yaml')
- `--mode`: Running mode - 'GRAPH' for static graph, 'PYNATIVE' for dynamic graph (default: 'GRAPH')
- `--device_target`: Computing platform - 'Ascend' or 'GPU' (default: 'Ascend')
- `--device_id`: Index of NPU or GPU (default: 0)

---

## Reference Resources

- [MindSpore Official Website](https://www.mindspore.cn/)
- [MindFlow Documentation](https://www.mindspore.cn/mindflow/)
- [3D Navier-Stokes Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes_3d/)
- [MindScience Repository](https://atomgit.com/mindspore-lab/mindscience)