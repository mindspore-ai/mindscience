---
name: fno2d
description: fno2d (Fourier Neural Operator for 2D) is a deep learning model that solves the 2-dimensional Navier-Stokes equation using neural operator learning. It learns mappings between infinite-dimensional function spaces to solve partial differential equations, providing fast inference for computational fluid dynamics tasks. Use this model when you need to predict solutions to 2D PDEs efficiently, particularly for fluid flow simulation.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team

---

# FNO2D

## Overview

FNO2D (Fourier Neural Operator for 2D) is a deep learning model that solves the 2-dimensional Navier-Stokes equation using the Fourier Neural Operator architecture. Unlike traditional neural networks that learn mappings between finite-dimensional spaces, FNO can learn mappings between infinite-dimensional function spaces, making it particularly effective for solving partial differential equations (PDEs).

The model consists of three main components:
- **Lifting Layer**: Lifts the input to a higher-dimensional representation
- **Fourier Layers**: Apply Fourier transform, perform linear transformations on lower Fourier modes, filter out higher modes, then apply inverse Fourier transform
- **Decoding Layer**: Decodes the final output from the high-dimensional representation

This skill provides inference capabilities for running FNO2D on Huawei Ascend NPUs using MindSpore.

---

## When to Use

- **2D PDE solving**: Solve 2-dimensional partial differential equations, particularly the Navier-Stokes equation
- **Computational fluid dynamics**: Fast prediction of fluid flow behavior without traditional numerical methods
- **Scientific computing**: Replace expensive finite element or finite difference methods with fast neural operator inference
- **Parametric PDE problems**: Learn operators that map initial conditions to solutions for a class of PDEs

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | NumPy arrays (.npy) containing initial vorticity and solution fields |
| Data Size   | Training: multiple samples with resolution 64x64 or higher; Inference: single or batch of initial conditions |
| Data Source | [Navier-Stokes Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/) |

#### Data Acquisition Methods

1. **Official MindSpore Dataset**: Download from [data_driven/navier_stokes/](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)
2. **Custom Data**: Generate initial vorticity w0(x) for the Navier-Stokes equation and prepare corresponding solution fields

#### Data Preprocessing

- Normalize input initial conditions to appropriate range
- Ensure input resolution matches model configuration (default 64x64)
- Prepare data in shape [batch_size, 1, height, width] for model input

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
3. **Download dataset**: Get the Navier-Stokes dataset from the official source
4. **Prepare config**: Configure model parameters in config file (default: `./configs/fno2d.yaml`)

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type         | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| Functional Limitations  | Solves Navier-Stokes in vorticity form for incompressible 2D flows |
| Performance Limitations | Inference time scales with resolution; higher resolutions take longer |
| Scale Limitations       | Default resolution is 64x64; custom resolutions require retraining |
| Input Format            | Must be 2D tensor with shape [batch, 1, H, W]               |

#### Notes

- **Note 1**: FNO2D learns the operator mapping w_t → w_(t+1) for the Navier-Stokes equation, enabling multi-step prediction by iterating inference
- **Note 2**: For NPU inference, ensure the Ascend CANN stack is properly installed and configured
- **Note 3**: The model uses Fourier transforms in the frequency domain, which requires FFT support in MindSpore
- **Note 4**: Configuration files (YAML) control model hyperparameters such as number of Fourier modes, layers, and training parameters

---

### 4. Model Invocation Guide

#### Model Initialization

```python
import mindspore as ms
from mindflow.cfd.flow_solver import FlowSolver
from mindflow.cfd.datasets import NavierStokes2D

# Set context for Ascend
ms.set_context(device_target="Ascend", device_id=0)

# Load configuration
config_path = "./configs/fno2d.yaml"
# Configuration should specify:
# - model: fno2d
# - resolution: 64
# - num_channels: 1
# - fno_modes: [12, 12]
# - num_layers: 4
```

#### Inference Execution

```python
# Initialize the model (using train.py from MindScience)
# The model can be run via command line:

export PYTHONPATH=$(cd ../../../../../ && pwd):$PYTHONPATH
python train.py --config_file_path ./configs/fno2d.yaml --mode GRAPH --device_target Ascend --device_id 0
```

Where:
- `--config_file_path`: Path to the parameter file (default: './configs/fno2d.yaml')
- `--mode`: Running mode - 'GRAPH' for static graph, 'PYNATIVE' for dynamic graph (default: 'GRAPH')
- `--device_target`: Computing platform - 'Ascend' or 'GPU' (default: 'Ascend')
- `--device_id`: Index of NPU or GPU (default: 0)

#### Result Post-processing

- Model outputs the predicted vorticity field w(x, t+1)
- Post-processing may include:
  - Converting vorticity to velocity field via stream function
  - Visualizing flow fields
  - Computing derived quantities (e.g., kinetic energy)

---

## Reference Resources

### Official Documentation

- [MindFlow FNO2D Documentation](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/fno2d/README_CN.md)
- [FNO Paper](https://arxiv.org/abs/2010.08895): Fourier Neural Operator for Parametric Partial Differential Equations

### Related Tutorials

- [MindSpore Official Website](https://www.mindspore.cn/)
- [MindFlow Documentation](https://www.mindspore.cn/mindflow)

### Community Support

- [MindSpore GitHub](https://github.com/mindspore-ai/mindspore)
- [MindScience Repository](https://atomgit.com/mindspore-lab/mindscience)