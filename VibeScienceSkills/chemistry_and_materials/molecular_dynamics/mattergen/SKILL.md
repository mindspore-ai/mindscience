---
name: mattergen
description: mattergen is a deep learning model for generative design of inorganic materials across the periodic table. It uses diffusion-based generative modeling to create novel crystal structures and can be fine-tuned to steer generation towards specific property constraints. Use this model when you need to generate new inorganic crystal structures, perform property-conditioned materials discovery, or explore the materials design space.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# MatterGen

## Overview

MatterGen is a state-of-the-art generative model for inorganic materials design developed by Microsoft's Materials Design Team at Microsoft Research AI for Science. It is a diffusion-based model that jointly predicts:

- Atomic fractional coordinates
- Element types
- Unit cell lattice vectors

The model enables researchers to generate novel inorganic material candidates with or without property constraints, accelerating materials discovery across the periodic table.

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits their task requirements.

- **Scenario 1**: Unconditional materials generation - Suitable for exploring novel inorganic crystal structures without any property constraints
- **Scenario 2**: Property-conditioned generation - Suitable for generating materials with specific target properties (band gap, magnetic density, bulk modulus, etc.)
- **Scenario 3**: Crystal structure prediction (CSP) - Suitable for generating structures conditioned on specific chemical compositions
- **Scenario 4**: High-throughput materials screening - Suitable for generating large candidate libraries for downstream property evaluation

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | CSV with structure data, CIF files, XYZ/extxyz files |
| Data Size | MP-20 (~45k structures), Alex-MP-20 (~600k structures) |
| Data Source | Materials Project, Alexandria database, or custom datasets |

#### Data Acquisition Methods

1. **MP-20 Dataset** - Download from MatterGen repository via Git LFS (~45k general inorganic materials with ≤20 atoms in unit cell)
2. **Alex-MP-20 Dataset** - Larger dataset combining MP-20 with Alexandria (stable structures <0.1 eV/atom above hull)
3. **Custom CIF Files** - Prepare crystal structure files using materials science software (VESTA, pymatgen, ASE)
4. **Hugging Face Hub** - Pre-trained model checkpoints available at https://huggingface.co/microsoft/mattergen

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Ensure crystal structures contain complete atomic positions and cell parameters
- **Step 2**: Verify element types are supported (excludes noble gases, radioactive elements, atomic number > 84)
- **Step 3**: For custom datasets, convert to CSV format with structure column
- **Step 4**: Use `csv-to-dataset` CLI tool to preprocess for training

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

| Component | Version |
| --------- | ----------------------------- |
| HDK       | 25.0.RC1 |
| CANN      | 8.3.RC1 |
| nnal      | 8.3.RC1 |
| Python    | 3.10 |
| torch     | 2.1.0 |
| torch-npu | 2.1.0 |

#### Clone repository and apply patch

**Note**: The path `Pytorch/matterGen` is case-sensitive on Linux.

```bash
git clone https://gitcode.com/AI4Science/MaterialsChemistry.git
cd MaterialsChemistry/Pytorch/matterGen
git clone https://github.com/microsoft/mattergen.git
cd mattergen
git checkout fcc7aee7fbf9c8d9932edc36f5e63a9a28314783
cp ../mattergen.patch .
git apply mattergen.patch
pip install -e .
```

#### Conda environment and Python dependencies

```bash
# Create conda environment
conda create --name mattergen_env python=3.10
conda activate mattergen_env

# Install uv package manager
pip install uv

# Install MatterGen in development mode
uv pip install -e .
```

**Git LFS Setup** (required for model checkpoints):
```bash
# Check if Git LFS is installed
git lfs --version

# If not installed:
sudo apt install git-lfs
git lfs install

# Pull specific checkpoint files
git lfs pull -I checkpoints/mattergen_base --exclude=""
```

#### Key Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| torch | 2.2.1+cu118 | PyTorch (Linux) or 2.4.1 (macOS) |
| pytorch-lightning | 2.0.6 | Training framework |
| torch_geometric | >=2.5 | Graph neural networks |
| pymatgen | >=2024.6.4 | Materials analysis |
| hydra-core | 1.3.1 | Configuration management |
| mattersim | >=1.1 | Property evaluation |

---

### 3. Running Inference

This module explains how to use the model for generating crystal structures.

#### Entry Point Scripts

| Command | Description |
|---------|-------------|
| `mattergen-generate` | Generate crystal structures |
| `mattergen-train` | Train MatterGen from scratch |
| `mattergen-finetune` | Fine-tune on property data |
| `mattergen-evaluate` | Evaluate generated structures |

#### Unconditional Generation

```bash
export MODEL_NAME=mattergen_base
export RESULTS_PATH=results/

# Generate samples
mattergen-generate $RESULTS_PATH \
    --pretrained-name=$MODEL_NAME \
    --batch_size=16 \
    --num_batches=1
```

#### Property-Conditioned Generation

```bash
# Single property conditioning (e.g., magnetic density)
export MODEL_NAME=dft_mag_density
mattergen-generate results/ \
    --pretrained-name=$MODEL_NAME \
    --batch_size=16 \
    --properties_to_condition_on="{'dft_mag_density': 0.15}" \
    --diffusion_guidance_factor=2.0
```

#### Multi-Property Conditioning

```bash
mattergen-generate results/ \
    --pretrained-name=chemical_system_energy_above_hull \
    --batch_size=16 \
    --properties_to_condition_on="{'energy_above_hull': 0.05, 'chemical_system': 'Li-O'}" \
    --diffusion_guidance_factor=2.0
```

#### Key Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--pretrained-name` | Pre-trained model name | None |
| `--model_path` | Path to custom checkpoint | None |
| `--batch_size` | Number of samples per batch | 64 |
| `--num_batches` | Number of batches to generate | 1 |
| `--properties_to_condition_on` | Property constraints (JSON dict) | {} |
| `--diffusion_guidance_factor` | Classifier-free guidance strength (γ) | 0.0 |
| `--record_trajectories` | Save denoising trajectories | True |
| `--sampling_config_name` | Sampling config (default/csp) | default |

#### Output Format

The generation script produces:

1. **`generated_crystals_cif.zip`**: ZIP file containing `.cif` files for each structure
2. **`generated_crystals.extxyz`**: Extended XYZ format with all structures
3. **`generated_trajectories.zip`** (if `--record_trajectories=True`): Full denoising trajectories

---

## Pre-trained Models

### Available Checkpoints

| Model Name | Description | Download |
|------------|-------------|----------|
| `mattergen_base` | Unconditional base model (Alex-MP-20) | Hugging Face |
| `mp_20_base` | Unconditional base model (MP-20) | Hugging Face |
| `chemical_system` | Conditioned on chemical system | Hugging Face |
| `space_group` | Conditioned on space group | Hugging Face |
| `dft_mag_density` | Conditioned on magnetic density | Hugging Face |
| `dft_band_gap` | Conditioned on band gap | Hugging Face |
| `ml_bulk_modulus` | Conditioned on bulk modulus | Hugging Face |
| `dft_mag_density_hhi_score` | Joint: magnetic density + HHI | Hugging Face |
| `chemical_system_energy_above_hull` | Joint: chemical system + stability | Hugging Face |

### Downloading Models

**Automatic (from Hugging Face):**
```python
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo

checkpoint_info = MatterGenCheckpointInfo.from_hf_hub("mattergen_base")
```

**Manual (via Git LFS):**
```bash
git lfs pull -I checkpoints/mattergen_base --exclude=""
```

**From Hugging Face Hub:**
```bash
# Models are available at: https://huggingface.co/microsoft/mattergen
```

---

## Data Format

### Input Data for Training

MatterGen expects crystal structures in CSV format with the following columns:

- `structure`: CIF string or structure data
- Optional property columns: `dft_band_gap`, `dft_mag_density`, `dft_bulk_modulus`, etc.

### Preprocessing Data

```bash
# Download and preprocess MP-20 dataset
git lfs pull -I data-release/mp-20/ --exclude=""
unzip data-release/mp-20/mp_20.zip -d datasets
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache
```

### Evaluation Input Formats

The evaluation script accepts:
- `.xyz` or `.extxyz` files
- `.zip` files containing `.cif` files
- Directories containing `.cif`, `.xyz`, or `.extxyz` files

### Training Datasets

1. **MP-20**: ~45k general inorganic materials (≤20 atoms in unit cell)
2. **Alex-MP-20**: ~600k structures from MP-20 + Alexandria (stable structures <0.1 eV/atom above hull)

---

## Hardware Requirements

| Resource | Requirement |
|----------|-------------|
| NPU | Huawei Ascend NPU (per GitCode MaterialsChemistry MatterGen README) |
| Memory | Sufficient host and device memory for training or generation batch sizes described in the official README |
| Disk | Space for datasets, checkpoints, and generated structures |

---

## Training

### Training Base Model

```bash
# Train on MP-20 (smaller dataset)
mattergen-train data_module=mp_20 ~trainer.logger

# Train on Alex-MP-20 (larger dataset)
mattergen-train data_module=alex_mp_20 ~trainer.logger trainer.accumulate_grad_batches=4
```

### Fine-tuning on Properties

```bash
# Single property fine-tuning
mattergen-finetune \
    adapter.pretrained_name=mattergen_base \
    data_module=mp_20 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
    ~trainer.logger \
    data_module.properties=["dft_mag_density"]
```

### Multi-Property Fine-tuning

```bash
mattergen-finetune \
    adapter.pretrained_name=mattergen_base \
    data_module=mp_20 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_mag_density=dft_mag_density \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.dft_band_gap=dft_band_gap \
    ~trainer.logger \
    data_module.properties=["dft_mag_density","dft_band_gap"]
```

---

## Evaluation

### Basic Evaluation

```bash
# Download reference dataset
git lfs pull -I data-release/alex-mp/reference_TRI2024correction.gz --exclude=""

# Evaluate with MatterSim relaxation
mattergen-evaluate \
    --structures_path=results/ \
    --relax=True \
    --structure_matcher='disordered' \
    --save_as="results/metrics.json" \
    --reference_dataset_path="data-release/alex-mp/reference_TRI2024correction.gz"
```

### Evaluation Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--structures_path` | Path to generated structures | Required |
| `--relax` | Relax structures with MatterSim | True |
| `--structure_matcher` | Structure matching (ordered/disordered) | disordered |
| `--reference_dataset_path` | Reference dataset for novelty check | None |
| `--energy_correction_scheme` | Energy correction (MP2020/TRI2024) | MP2020 |
| `--potential_load_path` | MatterSim model (1M/5M) | MatterSim-v1-1M |
| `--save_detailed_as` | Save per-structure metrics | None |

### Output Metrics

The evaluation produces:
- **S.U.N. %**: Stable, Novel, Unique structure percentage
- **Stability**: % of structures below energy threshold
- **Uniqueness**: % of unique structures
- **Novelty**: % of structures not in reference dataset
- **RMSD**: Average distance to relaxed structures

---

## License

**MIT License**

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Citation

If you use MatterGen in your research, please cite:

```bibtex
@article{MatterGen2025,
  author  = {Zeni, Claudio and Pinsler, Robert and Z{\"u}gner, Daniel and Fowler, Andrew and Horton, Matthew and Fu, Xiang and Wang, Zilong and Shysheya, Aliaksandra and Crabb{\'e}, Jonathan and Ueda, Shoko and Sordillo, Roberto and Sun, Lixin and Smith, Jake and Nguyen, Bichlien and Schulz, Hannes and Lewis, Sarah and Huang, Chin-Wei and Lu, Ziheng and Zhou, Yichi and Yang, Han and Hao, Hongxia and Li, Jielan and Yang, Chunlei and Li, Wenjie and Tomioka, Ryota and Xie, Tian},
  journal = {Nature},
  title   = {A generative model for inorganic materials design},
  year    = {2025},
  doi     = {10.1038/s41586-025-08628-5},
}
```

---

## Resources

- **GitCode MaterialsChemistry (MatterGen)**: https://gitcode.com/AI4Science/MaterialsChemistry/tree/main/Pytorch/matterGen
- **Official README**: https://gitcode.com/AI4Science/MaterialsChemistry/blob/main/Pytorch/matterGen/README.md
- **Additional reference**: https://github.com/microsoft/mattergen
- **Paper**: https://arxiv.org/abs/2312.03687
- **Hugging Face**: https://huggingface.co/microsoft/mattergen
- **MatterSim (for evaluation)**: https://github.com/microsoft/mattersim