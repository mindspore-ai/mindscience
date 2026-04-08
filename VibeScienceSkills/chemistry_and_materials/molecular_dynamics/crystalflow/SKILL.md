---
name: crystalflow
description: crystalflow is a flow-based deep learning model for crystal structure prediction. It uses neural ordinary differential equations and continuous density modeling with normalizing flows to generate plausible crystal structures. Use this model when you need to predict stable crystal structures from chemical composition for materials discovery.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team

---

# CrystalFlow

## Overview

CrystalFlow is a generative model for crystal structure prediction developed by MindSpore. It uses a flow-based architecture built on neural ordinary differential equations (Neural ODEs) and continuous density modeling with normalizing flows. Compared to diffusion-model-based generative methods, CrystalFlow is simpler, more flexible, and more efficient.

The model achieves competitive performance on benchmarks such as MP20 (Materials Project dataset with up to 20 atoms per unit cell) and can generate stable crystal structures for various chemical compositions.

This skill provides inference capabilities for CrystalFlow on Ascend NPU using MindSpore.

---

## When to Use

- **Crystal structure prediction**: Generate stable crystal structures from chemical composition
- **Materials discovery**: Explore plausible crystal structures for new materials
- **Property-guided generation**: Generate structures with specific target properties (when conditioned)
- **Multi-element systems**: Handle complex multi-element material systems

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | Crystal structure data in format compatible with MindChem (structure coordinates, lattice parameters) |
| Data Size   | MP20: ~15,000 stable structures; Perovskite: ~10,000; Carbon: ~2,000 |
| Data Source | MindSpore MindChem dataset repository |

#### Data Acquisition Methods

1. **MindSpore Dataset Download**: Download from https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/dataset/
2. **Supported Datasets**:
   - `perov_5`: Perovskite dataset
   - `carbon_24`: Carbon crystal dataset
   - `mp_20`: Materials Project dataset (up to 20 atoms/unit cell)
   - `mpts_52`: Materials Project dataset (up to 52 atoms/unit cell)

#### Data Preprocessing

- Download the dataset folders and `dataset_prop.txt` property file
- Place them under the `dataset` folder in the project directory
- Structure:
  ```
  crystalflow/
  └─dataset/
        perov_5/
        carbon_24/
        mp_20/
        mpts_52/
        dataset_prop.txt
  ```

---

### 2. Environment Configuration and Dependencies

#### Framework

CrystalFlow uses **MindSpore** (not PyTorch) as its deep learning framework.

#### Dependency Installation

```bash
# Clone MindChem repository
git clone https://gitcode.com/mindscience/MindChem.git
cd MindChem/applications/crystalflow

# Install dependencies
pip install -r requirement.txt
```

#### Environment Requirements

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Python Version | Python 3.8+                                   |
| Framework    | MindSpore                                      |
| Hardware     | Ascend NPU (910 series recommended)            |
| Memory       | At least 16GB RAM recommended                  |
| Disk Space   | At least 10GB for model and datasets           |

#### Installation Steps

1. **Step 1**: Clone the MindChem repository
2. **Step 2**: Navigate to `MindChem/applications/crystalflow`
3. **Step 3**: Download required datasets from the dataset link
4. **Step 4**: Install dependencies with `pip install -r requirement.txt`
5. **Step 5**: Configure the `config.yaml` file for inference parameters

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type         | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| Functional Limitations  | Generates crystal structures; requires composition as input |
| Performance Limitations | Generation quality depends on training data coverage        |
| Scale Limitations       | MP20 dataset supports up to 20 atoms per unit cell          |
| Input Format            | Requires chemical composition (element types and counts)    |

#### Notes

- **Note 1**: CrystalFlow is based on flow modeling (normalizing flows), not diffusion models
- **Note 2**: The model is trained on stable crystal structures from the Materials Project
- **Note 3**: For best results, use compositions that are well-represented in the training data
- **Note 4**: Generated structures may require additional relaxation/optimization for energy minimization

---

### 4. Model Invocation Guide

#### Model Initialization

The model weights are loaded automatically during inference. Configuration is managed through `config.yaml`.

#### Running Inference

From the CrystalFlow directory:

```bash
cd MindChem/applications/crystalflow

# Edit config.yaml to set test parameters
# Key settings:
#   - test.num_eval: number of samples to generate per composition
#   - test.checkpoint_path: path to model checkpoint

# Run inference
python evaluate.py
```

#### Configuration (config.yaml)

Key inference parameters in `config.yaml`:

```yaml
test:
  num_eval: 10          # Number of samples per composition
  checkpoint_path: ""   # Path to model checkpoint
  save_dir: "./results" # Output directory for generated structures
```

#### Result Post-processing

- Generated crystal structures are saved to the directory specified in `config.yaml`
- Output format: crystal structure files (coordinates, lattice parameters)
- Evaluation metrics can be computed using `compute_metric.py`

---

## Reference Resources

- **MindSpore MindChem**: https://gitee.com/mindspore/mindscience/tree/master/MindChem
- **CrystalFlow Repository**: MindChem/applications/crystalflow
- **Dataset Download**: https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/dataset/
- **Materials Project**: https://materialsproject.org/