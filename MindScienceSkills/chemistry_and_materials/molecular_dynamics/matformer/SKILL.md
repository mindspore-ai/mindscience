---
name: matformer
description: matformer is a state-of-the-art Graph Neural Network (GNN) combined with Transformer architecture for predicting properties of crystalline materials. It operates on periodic crystal graphs to capture both local and global structural information while remaining robust to lattice translations and other symmetries. Use this model when you need to predict formation energy per atom, bandgap, and lattice-related properties for 3D bulk crystalline materials.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team

---

# MatFormer

## Overview

MatFormer is a deep learning model based on Graph Neural Networks (GNNs) and the Transformer architecture, designed specifically for predicting properties of crystalline materials. By operating on periodic crystal graphs, MatFormer captures both local and global structural information while remaining robust to lattice translations and other symmetries.

Compared with classical models such as CGCNN, SchNet, and MEGNet, MatFormer achieves superior accuracy on tasks including formation energy per atom, bandgap, and lattice-related properties.

This skill provides inference capabilities for Ascend NPU using MindSpore, enabling users to run property prediction tasks on Huawei Ascend NPUs.

---

## When to Use

- **Formation energy prediction**: Predict formation energy per atom (eV/atom) for 3D crystalline materials
- **Bandgap prediction**: Predict electronic bandgap values using OptB88vdW, mBJ, or HSE06 functionals
- **Materials discovery**: Screen large databases of crystal structures for stable materials
- **Property prediction**: Predict lattice-related properties and material density
- **Convex hull analysis**: Calculate energy above convex hull (ehull) for stability assessment

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | JSON (jdft_3d-12-12-2022.json)                              |
| Data Size   | 75,993 3D bulk crystal structures                           |
| Data Source | JARVIS-DFT (Joint Automated Repository for Various Integrated Simulations) |

#### Data Acquisition Methods

1. **JARVIS-DFT Download**: Download `jdft_3d-12-12-2022.json` from https://figshare.com/articles/dataset/jdft_3d-7-7-2018_json/6815699 to the model directory without changing the file name.

#### Data Preprocessing

- The model automatically converts raw JARVIS data into graph representations during initialization
- Graph representations are cached under `dataset_dir` as configured in `config.yaml`
- Key data fields used:
  - `jid`: Unique JARVIS material ID (e.g., `JVASP-90856`)
  - `formula`: Chemical formula (e.g., `TiCuSiAs`)
  - `atoms`: Lattice matrix, atomic coordinates, and elements
  - `formation_energy_peratom`: Target property for prediction
  - `optb88vdw_bandgap`: Bandgap computed with OptB88vdW functional

---

### 2. Environment Configuration and Dependencies

#### Verified Ascend stack (reference)

| Component    | Version   |
| ------------ | --------- |
| HDK          | (per Huawei MindSpore installation) |
| CANN         | (per Huawei MindSpore installation) |
| Python       | 3.8+      |
| MindSpore    | >= 2.7.0  |
| MindScience  | (required for scientific computation) |

Install **MindSpore** and **MindScience** on the host per Huawei documentation:

- MindSpore installation: https://www.mindspore.cn/install
- MindScience installation: https://atomgit.com/mindspore-lab/mindscience

#### Clone LifeScience and prepare code

From a working directory of your choice:

```bash
git clone https://gitcode.com/AI4Science/LifeScience.git
cd LifeScience/MindSpore/applications/matformer
```

#### Conda environment and Python dependencies

```bash
conda create --name matformer python=3.8
conda activate matformer
pip install -r requirements.txt
```

#### Environment Requirements (hardware / disk)

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Hardware    | Ascend NPU (910 series recommended)            |
| Memory      | At least 16GB RAM recommended                  |
| Disk Space  | At least 10GB for model checkpoints and dataset |

#### End-to-end checklist

| Step | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 1    | Clone **LifeScience**, enter `MindSpore/applications/matformer`. |
| 2    | Install MindSpore >= 2.7.0 and MindScience per Huawei docs; create conda env `matformer` (Python 3.8+); `pip install -r requirements.txt`. |
| 3    | **Data**: Download `jdft_3d-12-12-2022.json` from figshare to the `matformer` directory. |
| 4    | **Weights**: Ensure checkpoint exists at path specified in `config.yaml` (default `./ckpt/best_matformer.ckpt`). |
| 5    | **Inference**: Run `python predict.py` from the `matformer` directory. |

Use Linux or WSL (or Git Bash on Windows) so `bash` is available.

Optional NPU check:

```bash
python -c "import mindspore; print(mindspore.__version__); print(mindspore.get_context('device_target'))"
```

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type         | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| Functional Limitations  | Requires valid crystal structure data; cannot predict de novo structures |
| Performance Limitations | Inference time scales with number of atoms in structure; larger crystals take longer |
| Scale Limitations       | Optimized for 3D bulk crystals; not suitable for 2D materials or molecules |
| Input Format            | Must be valid JARVIS-DFT JSON format with complete structural data |

#### Notes

- **Note 1**: MatFormer is designed for periodic 3D crystal structures. Ensure input data contains valid lattice matrices and atomic coordinates.
- **Note 2**: For NPU inference, install the **MindSpore** stack (>= 2.7.0) and **MindScience** per Huawei documentation and verify the NPU is visible to MindSpore.
- **Note 3**: Prediction properties are configured in `config.yaml` under the `predictor` section. Common targets include `formation_energy_peratom`, `optb88vdw_bandgap`, `mbj_bandgap`, and `hse_gap`.
- **Note 4**: Checkpoint path is specified in `predictor.checkpoint_path` field of `config.yaml` (default `./ckpt/best_matformer.ckpt`).

---

### 4. Model Invocation Guide

#### Model Initialization

The model uses MindSpore for inference. Checkpoint files are saved during training and loaded during inference.

| Property Type | Description |
| ------------- | ------------------------------------------------------------ |
| formation_energy_peratom | Formation energy per atom (eV/atom) |
| optb88vdw_bandgap | Bandgap with OptB88vdW functional (eV) |
| mbj_bandgap | Bandgap with mBJ functional (eV) |
| hse_gap | Bandgap with HSE06 functional (eV) |
| density | Material density (g/cm³) |
| ehull | Energy above convex hull (eV/atom) |

#### Running Inference

**Shell (recommended):** run from the `matformer` directory (same directory as `predict.py` and `config.yaml`).

```bash
cd /path/to/LifeScience/MindSpore/applications/matformer
python predict.py
```

**Configuration:** Edit `config.yaml` to set:
- `predictor.checkpoint_path`: Path to the trained checkpoint (default `./ckpt/best_matformer.ckpt`)
- `predictor.props`: Target property to predict (e.g., `formation_energy_peratom`)
- `predictor.epoch_size`: Number of inference epochs

#### Result Post-processing

Prediction results are printed in the logs and can be saved or post-processed as needed. The model outputs:
- Predicted property values (e.g., formation energy per atom in eV/atom)
- Comparison with ground truth values if available in the dataset

---

## Reference Resources

- **Ascend / LifeScience (this skill)**: https://gitcode.com/AI4Science/LifeScience/tree/main/MindSpore/applications/matformer
- **MatFormer Paper**: https://arxiv.org/abs/2209.11807
- **JARVIS-DFT**: https://jarvis.nist.gov/
- **MindSpore Installation**: https://www.mindspore.cn/install
- **MindScience**: https://atomgit.com/mindspore-lab/mindscience