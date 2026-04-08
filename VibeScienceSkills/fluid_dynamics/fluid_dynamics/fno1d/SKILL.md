---
name: fno1d
description: fno1d (Fourier Neural Operator for 1D) is a deep learning model that solves the 1-dimensional Burgers equation using neural operator learning. It learns mappings between infinite-dimensional function spaces to solve partial differential equations, providing fast inference for computational fluid dynamics tasks. Use this model when you need to predict solutions to 1D PDEs efficiently.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team

---

# FNO1D

## Overview

FNO1D (Fourier Neural Operator for 1D) is a deep learning model that solves the 1-dimensional Burgers equation using the Fourier Neural Operator architecture. Unlike traditional neural networks that learn mappings between finite-dimensional spaces, FNO can learn mappings between infinite-dimensional function spaces, making it particularly effective for solving partial differential equations (PDEs).

The model consists of three main components:
- **Lifting Layer**: Lifts the input to a higher-dimensional representation
- **Fourier Layers**: Apply Fourier transform, perform linear transformations on lower Fourier modes, filter out higher modes, then apply inverse Fourier transform
- **Decoding Layer**: Decodes the final output from the high-dimensional representation

This skill provides inference capabilities for running FNO1D on Huawei Ascend NPUs using MindSpore.

---

## When to Use

- **1D PDE solving**: Solve 1-dimensional partial differential equations, particularly the Burgers equation
- **Computational fluid dynamics**: Fast prediction of fluid flow behavior without traditional numerical methods
- **Scientific computing**: Replace expensive finite element or finite difference methods with fast neural operator inference
- **Parametric PDE problems**: Learn operators that map initial conditions to solutions for a class of PDEs

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | NumPy arrays (.npy) containing initial conditions and solution fields |
| Data Size   | Training: multiple samples with resolution 1024; Inference: single or batch of initial conditions |
| Data Source | [1D Burgers Equation Resolution Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/) |

#### Data Acquisition Methods

1. **Official MindSpore Dataset**: Download from [data_driven/burgers/](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/)
2. **Custom Data**: Generate initial conditions u0(x) for the Burgers equation and prepare corresponding solution fields u(x, t=1)

#### Data Preprocessing

- Normalize input initial conditions to appropriate range
- Ensure input resolution matches model configuration (default 1024)
- Prepare data in shape [batch_size, 1, resolution] for model input

---

### 2. Environment Configuration and Dependencies

#### Verified Ascend stack (reference)

| Component | Version     |
| --------- | ----------- |
| HDK       | [Per MindSpore requirements] |
| CANN      | [Per MindSpore requirements] |
| Python    | 3.8+        |
| MindSpore | 2.7.0       |

#### Environment Setup

```bash
# Install MindSpore
pip install mindspore==2.7.0

# Install MindFlow (if required)
pip install mindflow
```

#### Environment Requirements (hardware / disk)

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Python Version | 3.8+                                           |
| Hardware    | Ascend 32G or GPU with CUDA support            |
| Memory      | At least 16GB RAM                              |
| Disk Space  | ~500MB for model and dataset                   |

#### Installation Steps

1. **Install MindSpore**: Follow official MindSpore installation guide for Ascend
2. **Install dependencies**: Install required Python packages
3. **Download dataset**: Get the Burgers equation dataset from the official source
4. **Prepare config**: Configure model parameters in config file (default: `./configs/fno1d.yaml`)

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
|----------------|--------------|
| Functional Limitations | Currently supports 1D Burgers equation; other PDEs require model retraining |
| Performance Limitations | Optimized for resolution=1024; other resolutions may require adjustment |
| Scale Limitations | Input must match training resolution (1024 by default) |

#### Notes

- **Note 1**: The model maps initial conditions u0(x) to solution u(x, t=1) for the Burgers equation
- **Note 2**: Default configuration uses modes=16, hidden_channels=64, depth=4
- **Note 3**: Supports both Ascend NPU and GPU execution
- **Note 4**: Can run in GRAPH (static) or PYNATIVE (dynamic) mode

---

### 4. Model Invocation Guide

#### Model Configuration

The model uses a YAML configuration file (default: `./configs/fno1d.yaml`):

```yaml
# Example configuration
model:
  name: fno1d
  resolution: 1024
  modes: 16
  hidden_channels: 64
  depth: 4

training:
  batch_size: 8
  epoch: 100

optimizer:
  name: Adam
```

#### Inference Code

```python
import mindspore as ms
from mindflow import load_yaml_config
from train import FNO1D

# Load configuration
config = load_yaml_config('./configs/fno1d.yaml')

# Initialize model
model = FNO1D(
    resolution=config['model']['resolution'],
    modes=config['model']['modes'],
    hidden_channels=config['model']['hidden_channels'],
    depth=config['model']['depth']
)

# Load pretrained weights
model.load_weights('./checkpoints/fno1d.ckpt')

# Prepare input (initial condition u0)
# Shape: [batch_size, 1, resolution]
input_data = ms.Tensor(initial_conditions, ms.float32)

# Run inference
output = model(input_data)
# Output shape: [batch_size, 1, resolution] - solution at t=1
```

#### Running from Command Line

```bash
export PYTHONPATH=$(cd ../../../../../ && pwd):$PYTHONPATH
python train.py --config_file_path ./configs/fno1d.yaml --device_target Ascend --device_id 0 --mode GRAPH
```

Parameters:
- `--config_file_path`: Path to parameter file (default: './configs/fno1d.yaml')
- `--device_target`: Computing platform - 'Ascend' or 'GPU' (default: 'Ascend')
- `--device_id`: Index of NPU or GPU (default: 0)
- `--mode`: Running mode - 'GRAPH' (static) or 'PYNATIVE' (dynamic)

---

### 5. Performance Metrics

| Metric | Value |
|--------|-------|
| Parameters | 5.57e5 |
| Train Loss (MSE) | 0.004872 |
| Evaluation Error (RMSE) | 0.001088 |
| Speed | ~15 ms/step |

#### Default Configuration

| Parameter | Value |
|-----------|-------|
| Resolution | 1024 |
| Modes | 16 |
| Hidden Channels | 64 |
| Depth | 4 |
| Batch Size | 8 |

---

## Reference Resources

- **Paper**: [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- **MindSpore Documentation**: https://www.mindspore.cn/
- **MindFlow Documentation**: https://www.mindspore.cn/mindflow/
- **Dataset Download**: [1D Burgers Equation Resolution Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/)
- **Jupyter Notebook**: [English Version](./FNO1D.ipynb), [Chinese Version](./FNO1D_CN.ipynb)

---

## Contributor

- gitee id: [liulei277](https://gitee.com/liulei277), [yezhenghao2023](https://gitee.com/yezhenghao2023), [huangwangwen2025](https://gitee.com/huangwangwen2025)
- email: liulei2770919@163.com, yezhenghao@isrc.iscas.ac.cn, wangwen@isrc.iscas.ac.cn