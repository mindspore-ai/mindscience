---
name: p2c2net
description: p2c2net (PDE-Preserved Coarse Correction Network) is a novel neural network architecture designed to efficiently solve spatiotemporal partial differential equations (PDEs) on coarse mesh grids with limited training data. Use this model when you need to solve 2D Burgers equation or similar spatiotemporal PDEs on Ascend NPU.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend

---

# P2C2Net

## Overview

P2C2Net (PDE-Preserved Coarse Correction Network) is a neural network architecture for efficiently solving spatiotemporal partial differential equations (PDEs) on coarse mesh grids with limited training data. The model consists of two synergistic modules: (1) a trainable PDE block that learns to update the coarse solution based on a high-order numerical scheme with boundary condition encoding, and (2) a neural network block that consistently corrects the solution on the fly. The model adopts a learnable symmetric Conv filter, with weights shared over the entire model, to accurately estimate the spatial derivatives of PDE based on the neural-corrected system state.

This skill provides inference capabilities for solving 2D Burgers equation on Huawei Ascend NPUs using MindSpore.

---

## When to Use

### Hardware Requirements

This model requires Ascend hardware. Before running, please verify that your device is Ascend:

```python
import subprocess

def check_npu_device():
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("Ascend not detected. This model requires Ascend hardware.")
    except FileNotFoundError:
        raise RuntimeError("npu-smi command not found. Please ensure Ascend driver is installed.")

check_npu_device()
```

- **2D Burgers equation solving**: Solve the 2D Burgers' equation efficiently using neural network-based PDE solving
- **Spatiotemporal dynamics prediction**: Predict spatiotemporal dynamics on coarse mesh grids
- **Limited data PDE solving**: Solve PDEs when training data is limited
- **Periodic boundary conditions**: Handle problems with periodic boundary conditions

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | NumPy arrays (.npy) containing 2D velocity fields |
| Data Size | Training and testing datasets generated via dataGen.py |
| Data Source | Generated from MindFlow/applications/data_mechanism_fusion/p2c2net/src/dataGen.py |

#### Data Acquisition Methods

1. **Generate from source**: Download dataGen.py from MindFlow repository and run to generate training and testing data
2. **Pre-generated datasets**: Use pre-generated datasets if available from the repository

#### Data Preprocessing

- Run `python dataGen.py` under the `src` directory to generate training and testing datasets
- The data generation script creates synthetic 2D Burgers equation solutions with periodic boundary conditions

---

### 2. Environment Configuration and Dependencies

#### Verified Ascend Stack (reference)

| Component | Version |
| --------- | ----------- |
| CANN      | 8.2.RC1     |
| Python    | 3.10+       |
| MindSpore | >=2.5.0     |
| MindScience | 0.8.0     |

#### Installation Steps

1. **Install MindSpore and MindScience**: Ensure the correct versions are installed
   ```bash
   pip install mindspore>=2.5.0
   pip install mindscience==0.8.0
   ```

2. **Install additional dependencies**:
   ```bash
   pip install numpy pandas sympy matplotlib
   ```

3. **Obtain source code**: Clone the MindScience repository or obtain codes directly from [MindFlow/applications/data_mechanism_fusion/p2c2net](https://atomgit.com/mindspore-lab/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/p2c2net)

#### Environment Requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ---------------------------------------------- |
| NPU         | Ascend NPU with memory >32GB                   |
| Python      | 3.10+                                          |
| Disk Space  | Sufficient for datasets and model checkpoints |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
|----------------|--------------|
| Functional Limitations | Currently supports 2D Burgers equation; other PDEs require modifications |
| Performance Limitations | Accuracy depends on training data quality and quantity |
| Scale Limitations | Optimized for coarse mesh grids; fine grids may require more memory |

#### Notes

- **Note 1**: Periodic boundary conditions are used to avoid non-physical reflections/errors
- **Note 2**: The model uses learnable symmetric Conv filters for spatial derivative estimation
- **Note 3**: Training requires both PDE block and neural network block working synergistically

---

### 4. Model Invocation Guide

#### Dataset Generation

First, generate the training and testing data:

```bash
cd src
python dataGen.py
```

#### Running Inference/Prediction

After training, use the trained model to solve the 2D Burgers equation:

```bash
python p2c2net/train_burgers.py --experiment p2c2net
```

**Command-line arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--experiment` | Experiment directory containing config files | Required |
| `--mode` | Running mode: 'GRAPH' (static) or 'PYNATIVE' (dynamic) | 'GRAPH' |
| `--device_target` | Computing platform: 'Ascend' or 'GPU' | 'Ascend' |
| `--device_id` | NPU device ID | 0 |
| `--config_filename` | Configuration file name under configs/ | 'burgers.json' |
| `--train_stage` | Enable training mode | True |
| `--test_stage` | Enable testing mode | True |
| `--continue` | Resume training from saved checkpoint | False |

#### Configuration

Experiment configurations are stored under the `config/` directory. The default configuration file is `burgers.json` which defines model parameters and training schedule.

#### Results

After running, experiment outputs (checkpoints and evaluation results) are saved in the result directory under the `--experiment` directory provided.

---

## Reference Resources

- **Original Paper**: [P2C2Net: PDE-Preserved Coarse Correction Network for Efficient Prediction of Spatiotemporal Dynamics](https://arxiv.org/pdf/2411.00040)
- **MindSpore Documentation**: https://www.mindspore.cn/
- **MindScience Repository**: https://atomgit.com/mindspore-lab/mindscience
- **Model Code**: MindFlow/applications/data_mechanism_fusion/p2c2net

---

## License

Apache License 2.0 - https://atomgit.com/mindspore-lab/mindscience/blob/master/LICENSE

---

## Citation

If this project is helpful for your research, please cite:

Wang Q, Ren P, Zhou H, et al. P²C²Net: PDE-preserved coarse correction network for efficient prediction of spatiotemporal dynamics[C]//The Thirty-eighth Annual Conference on Neural Information Processing Systems. 2024.