---
name: diffcsp
description: diffcsp is a diffusion-based deep generative framework for crystal structure prediction. It reformulates the search for stable crystal structures as a generative task, using a periodic E(3)-equivariant graph neural network to directly generate plausible 3D atomic structures (lattice and atomic coordinates) from chemical composition. Use this model when you need to predict crystal structures from chemical formulas.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team

---

# DiffCSP

## Overview

DiffCSP is a diffusion-based deep generative framework for crystal structure prediction developed by Huawei MindSpore. It reformulates the search for stable crystal structures as a generative task by learning the distribution of large-scale crystal datasets. The model can directly and efficiently generate plausible 3D atomic structures including lattice parameters and atomic coordinates from only the chemical composition (atom types and ratios).

The key innovation of DiffCSP lies in using a periodic E(3)-equivariant graph neural network that explicitly incorporates translational, rotational, and periodic symmetries. This ensures that generated structures strictly obey physical constraints, enabling efficient exploration of the crystal configuration space and producing high-quality candidates at a much lower computational cost than first-principles methods.

This skill provides inference capabilities for Ascend NPU using MindSpore, enabling users to generate crystal structures from chemical compositions.

---

## When to Use

- **Crystal structure prediction**: Generate 3D crystal structures from chemical formulas
- **Materials discovery**: Explore novel crystal configurations for materials design
- **Computational materials science**: Accelerate crystal structure exploration without expensive quantum-mechanical calculations
- **Property-guided generation**: Generate structures that may exhibit specific material properties

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | Crystal structure data in format compatible with the dataset loader |
| Data Size   | Pre-trained models available; datasets include Perovskite (perov_5), Carbon (carbon_24), MP-20, MPTS-52 |
| Data Source | Download from https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/dataset/ |

#### Data Acquisition Methods

1. **Pre-trained Models**: Download pretrained checkpoints from https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/pre-train
2. **Training Data**: Download dataset folders and `dataset_prop.txt` from the dataset link above

#### Data Preprocessing

- Place dataset folders under `dataset/` directory in the project root
- Ensure `dataset_prop.txt` is in the dataset directory
- Supported datasets: perov_5 (Perovskite), carbon_24 (Carbon crystal), mp_20 (MP up to 20 atoms), mpts_52 (MP up to 52 atoms)

---

### 2. Environment Configuration and Dependencies

#### Dependency Installation

```bash
# Install MindSpore (see official guide: https://www.mindspore.cn/install)
# Install MindScience (see https://atomgit.com/mindspore-lab/mindscience)

# Install Python dependencies
pip install -r requirement.txt
```

#### Environment Requirements

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Python Version | Python 3.8+ (MindSpore >= 2.7.0)            |
| Framework    | MindSpore >= 2.7.0, MindScience >= 0.8.0    |
| Hardware     | Ascend NPU (910 series recommended)           |
| Memory       | At least 16GB RAM recommended                 |
| Disk Space   | At least 10GB for model checkpoints and datasets |

#### Installation Steps

1. **Install MindSpore**: Follow the official guide at https://www.mindspore.cn/install
2. **Install MindScience**: See https://atomgit.com/mindspore-lab/mindscience
3. **Install dependencies**: Run `pip install -r requirement.txt`
4. **Prepare dataset**: Download and place datasets under `dataset/` directory

#### End-to-end checklist

| Step | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 1    | Install MindSpore >= 2.7.0 and MindScience >= 0.8.0        |
| 2    | Clone or download DiffCSP application code                  |
| 3    | Run `pip install -r requirement.txt`                        |
| 4    | Download pretrained checkpoints from the dataset link       |
| 5    | Configure `config.yaml` with checkpoint path and inference parameters |
| 6    | Run inference: `python evaluate.py`                         |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type         | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| Functional Limitations  | Generates crystal structures from composition; requires known chemical formula |
| Performance Limitations | Number of generated samples affects quality (controlled by `test.num_eval`) |
| Scale Limitations       | Dataset-specific limits on atoms per unit cell              |
| Input Format            | Chemical composition (atom types and ratios)                |

#### Notes

- **Note 1**: DiffCSP uses periodic E(3)-equivariant graph neural networks to ensure physical symmetry constraints
- **Note 2**: For inference, set `checkpoint.last_path` in `config.yaml` to the pretrained model path
- **Note 3**: The `test.num_eval` parameter controls how many samples are generated per composition
- **Note 4**: Generated crystals are saved as Python dictionaries with structure and ground truth data

---

### 4. Model Invocation Guide

#### Model Initialization

Configure `config.yaml` with the following key settings:

```yaml
checkpoint:
  last_path: /path/to/pretrained/checkpoint.ckpt

test:
  num_eval: 10  # Number of samples to generate per composition
  eval_save_path: ./eval_results.pkl
```

#### Running Inference

From the `diffcsp` directory:

```bash
python evaluate.py
```

The inference script loads the pretrained model and generates crystal structures based on the compositions defined in the configuration.

#### Result Post-processing

Generated crystals are saved to the file specified by `test.eval_save_path`. The output is a Python dictionary:

```python
{
    'pred': [
        [crystal_A sample_1, crystal_A sample_2, ..., crystal_A sample_n],
        [crystal_B sample_1, crystal_B sample_2, ..., crystal_B sample_n]
    ],
    'gt': [
        crystal_A ground_truth,
        crystal_B ground_truth,
        ...
    ]
}
```

To evaluate the generated structures:

```bash
python compute_metric.py
```

Evaluation results are saved as JSON files with metrics like:
```json
{"match_rate": 0.986, "rms_dist": 0.013}
```

---

## Reference Resources

- **MindSpore Official**: https://www.mindspore.cn/
- **MindScience**: https://atomgit.com/mindspore-lab/mindscience
- **DiffCSP Dataset**: https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/dataset/
- **DiffCSP Pretrained Models**: https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/pre-train
- **Paper**: Jiao R, Huang W, Lin P, et al. Crystal structure prediction by joint equivariant diffusion[J]. Advances in Neural Information Processing Systems, 2024, 36.