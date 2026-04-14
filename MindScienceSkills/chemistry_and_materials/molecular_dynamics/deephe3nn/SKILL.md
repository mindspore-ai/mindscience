---
name: deephe3nn
description: deephe3nn is an E(3)-equivariant neural network for accurately predicting the electronic Hamiltonian of a system from the atomic configuration in crystals. Use this model when you need to predict electronic Hamiltonians for crystal structures (e.g., bilayer graphene) for downstream tasks such as band structure and transport-property calculations.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team

---

# DeephE3nn

## Overview

DeephE3nn is an E(3)-equivariant neural network that predicts the electronic Hamiltonian of a system from atomic configurations in crystals. By explicitly modeling rotational and translational symmetries in space via an equivariant graph neural network, DeephE3nn efficiently learns the mapping from "crystal structure → electronic Hamiltonian". It significantly reduces computational cost while preserving physical symmetries.

This skill provides inference capabilities for predicting electronic Hamiltonians of material systems, such as bilayer graphene, enabling efficient approximations for downstream tasks including band structure and transport-property calculations.

---

## When to Use

- **Electronic Hamiltonian prediction**: Predict electronic Hamiltonians from crystal structures for materials discovery
- **Band structure calculations**: Use predicted Hamiltonians for band structure computations
- **Transport property simulations**: Calculate transport properties based on predicted electronic structures
- **High-throughput materials screening**: Efficiently screen large numbers of crystal structures

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | Crystal structure data in the Bilayer_graphene_dataset format (extracted from `Bilayer_graphene_dataset.zip`) |
| Data Size   | Dataset contains bilayer graphene structures with varying configurations |
| Data Source | Download from Zenodo: https://zenodo.org/records/7553640 |

#### Data Acquisition Methods

1. **Zenodo Download**: Download `Bilayer_graphene_dataset.zip` from https://zenodo.org/records/7553640
2. **Extract**: Unzip the file to the current directory without changing the file name

#### Data Preprocessing

- Ensure `Bilayer_graphene_dataset` directory is extracted to the project root
- The dataset should be placed alongside the `configs` directory containing `Bilayer_graphene_train.ini`
- Verify the directory structure matches:
  ```
  deephe3nn
      ├─Bilayer_graphene_dataset
      └─configs
             Bilayer_graphene_train.ini
  ```

---

### 2. Environment Configuration and Dependencies

#### Environment Requirements

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Python Version | Python 3.x (recommended)                    |
| Framework    | MindSpore >= 2.7.0, MindScience >= 0.8.0     |
| Hardware     | Ascend NPU (specify device via `-device_id`) |
| Memory       | Sufficient memory for graph neural network operations |
| Disk Space   | At least 2GB for dataset and model checkpoints |

#### Dependency Installation

1. **Install MindSpore**: Follow the official guide at https://www.mindspore.cn/install
2. **Install MindScience**: See https://atomgit.com/mindspore-lab/mindscience
3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

#### Installation Steps

1. **Step 1**: Install MindSpore >= 2.7.0 per official documentation
2. **Step 2**: Install MindScience >= 0.8.0 (provides `mindscience.e3nn` and related modules)
3. **Step 3**: Install requirements: `pip install -r requirements.txt`
4. **Step 4**: Download and extract the dataset

#### End-to-end checklist

| Step | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 1    | Install MindSpore >= 2.7.0 and MindScience >= 0.8.0        |
| 2    | Clone or obtain the DeephE3nn repository                    |
| 3    | Run `pip install -r requirements.txt`                       |
| 4    | Download `Bilayer_graphene_dataset.zip` from Zenodo and extract |
| 5    | **Inference**: Run `python predict.py configs/Bilayer_graphene_train.ini` |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type         | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| Functional Limitations  | Predicts electronic Hamiltonian; downstream property calculations depend on external tools |
| Performance Limitations | Computational cost scales with system size                  |
| Scale Limitations       | Optimized for crystal structures similar to bilayer graphene |
| Input Format            | Requires properly formatted crystal structure data          |

#### Notes

- **Note 1**: DeephE3nn uses MindSpore as the backend framework, not PyTorch
- **Note 2**: The model explicitly models E(3) equivariance (rotational and translational symmetries)
- **Note 3**: For inference, set the checkpoint path in the `checkpoint_dir` field of the config file before running
- **Note 4**: The `-device_id` argument can be used to specify the Ascend device ID

---

### 4. Model Invocation Guide

#### Running Inference

From the `deephe3nn` directory, run:

```bash
python predict.py configs/Bilayer_graphene_train.ini
```

To specify a device ID:

```bash
python predict.py configs/Bilayer_graphene_train.ini -device_id 0
```

#### Configuration

The inference uses the configuration file `configs/Bilayer_graphene_train.ini`. Key settings include:
- `checkpoint_dir`: Path to the trained model checkpoint
- Other model and data parameters as defined in the config

#### Result Post-processing

The inference script computes the electronic Hamiltonian for the given structures and prints evaluation results in the logs. The exact output format can be adjusted through the configuration and downstream task requirements.

---

## Reference Resources

- **MindSpore Installation**: https://www.mindspore.cn/install
- **MindScience**: https://atomgit.com/mindspore-lab/mindscience
- **Dataset Source**: https://zenodo.org/records/7553640
- **DeephE3nn Paper**: Xiaoxun Gong, He Li, Nianlong Zou, et al. General framework for E(3)-equivariant neural network representation of density functional theory Hamiltonian[J]. Nature Communications, 2023, 14: 2848.