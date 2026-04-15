---
name: genmol
description: genmol is a masked discrete diffusion model for molecular generation developed by NVIDIA Digital Bio. Use this model when you need to generate novel drug-like molecules, design molecular linkers, extend molecular motifs, or perform scaffold decoration and lead optimization for drug discovery.
license: Apache 2.0 (code) / NVIDIA Open Model License (weights)
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend
---

# GenMol

## Overview

GenMol (Generative Molecule) is a drug discovery generalist model developed by NVIDIA Digital Bio. It is a masked discrete diffusion model trained on molecular Sequential Attachment-based Fragment Embedding (SAFE) representations for fragment-based molecule generation. The model uses a BERT-based Transformer architecture with discrete diffusion to generate high-quality, drug-like molecules.

GenMol supports multiple generation tasks including de novo generation, fragment-constrained generation, linker design, motif extension, scaffold decoration/morphing, superstructure generation, goal-directed hit generation, and lead optimization.

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

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits their task requirements.

- **Scenario 1**: De novo molecule generation - Suitable for generating novel drug-like molecules from scratch
- **Scenario 2**: Fragment-constrained generation - Suitable for generating molecules that incorporate specific molecular fragments
- **Scenario 3**: Linker design - Suitable for connecting molecular fragments with appropriate linkers
- **Scenario 4**: Scaffold decoration/morphing - Suitable for modifying molecular scaffolds while preserving core structure
- **Scenario 5**: Goal-directed hit generation - Suitable for generating molecules optimized for specific properties (e.g., QED, synthetic accessibility)
- **Scenario 6**: Lead optimization - Suitable for optimizing existing molecules for desired properties while maintaining similarity

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Input Format | SMILES strings for molecules, or SDF/PDB for fragment-based generation |
| SAFE Representation | Molecular fragments encoded in Sequential Attachment-based Fragment Embedding format |
| Data Size | Single molecules or batch files; no minimum size requirement |
| Data Source | User-provided molecules, or use built-in oracle functions for goal-directed generation |

#### Data Acquisition Methods

1. **User-provided SMILES** - Provide SMILES strings directly for generation or optimization
2. **Fragment Files** - Provide molecular fragments as SDF/PDB files for constrained generation
3. **Built-in Oracles** - Use PMO benchmark or custom oracle functions for property optimization

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Ensure input molecules are in valid SMILES or SDF format
- **Step 2**: For fragment-constrained generation, prepare fragment structures in separate files
- **Step 3**: Verify molecule validity using RDKit before passing to GenMol
- **Step 4**: Place checkpoint file (`model.ckpt` or `model_v2.ckpt`) in the `checkpoints/` directory

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

| Component | Version (per official README) |
| --------- | ---------------------------------------- |
| HDK       | 25.0.rc1.1 |
| CANN      | 8.2.RC1 |
| Python    | 3.10 |
| torch     | 2.6.0 |
| torch-npu | 2.6.0 |
| transformers | 4.52.4 |

#### Clone repository

```bash
git clone https://gitcode.com/AI4Science/GenMol.git
cd GenMol
bash env/setup.sh
```

#### Conda environment and Python dependencies

```bash
# Create conda environment
conda create --name genmol python=3.10
conda activate genmol

# Install dependencies using the setup script
bash env/setup.sh

# Or manually:
pip install -r env/requirements.txt
pip install -e .

# Additional dependency required for hit generation (gsk3b, jnk3)
pip install scikit-learn==1.2.2
```

**Key dependencies:**
- `torch==2.6.0`
- `transformers==4.56.2`
- `lightning==2.5.1`
- `hydra-core==1.3.2`
- `safe-mol==0.1.14`
- `bionemo-moco>=0.0.2.1`

#### Download model parameters

GenMol model weights are available from NVIDIA NGC:

- **GenMol V1**: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/genmol_v1
- **GenMol V2**: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/genmol_v2

```bash
# Download checkpoint and place in checkpoints directory
# Rename to model.ckpt for V1 or model_v2.ckpt for V2
mv downloaded_checkpoint.ckpt checkpoints/model.ckpt
```

Update `./configs/base.yaml` with the correct checkpoint path if needed.

#### Environment Requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend NPU (per GitCode README) |
| Memory | Large checkpoints benefit from ample device memory (see README) |
| Disk Space | ~2GB for model checkpoints and dependencies |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | Clone: `git clone https://gitcode.com/AI4Science/GenMol.git` |
| 2 | Create conda env: `conda create --name genmol python=3.10` |
| 3 | Install dependencies: `bash env/setup.sh` |
| 4 | Download model checkpoint from NVIDIA NGC |
| 5 | Place checkpoint in `checkpoints/` directory |
| 6 | Run inference: `python scripts/exps/denovo/run.py` |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Hardware Limitation | Follow **https://gitcode.com/AI4Science/GenMol** README for the supported Ascend stack |
| Input Length | Maximum 256 tokens in SAFE representation |
| Molecule Size | Optimized for drug-like molecules (MW < 500 Da) |
| Generation Speed | Slower than single-step generative models due to iterative diffusion |
| Oracle Dependencies | Goal-directed generation requires specific oracle packages (pytdc) |

#### Notes

- **Note 1**: GenMol V2 uses extended SAFE syntax with angle-brackets for inter-fragment attachment points, providing better performance in fragment-constrained generation.
- **Note 2**: The model uses masked discrete diffusion - generation is iterative and may take several seconds per molecule.
- **Note 3**: For goal-directed generation, ensure `pytdc` is installed for oracle functions (gsk3b, jnk3, qed, sa).
- **Note 4**: Model weights are governed by NVIDIA Open Model License - commercial use is permitted but check license terms.

---

### 4. Model Invocation Guide

#### Model Configuration

| Item | Example / value |
| ---- | ---------------- |
| Model Type | Masked Discrete Diffusion (BERT-based Transformer) |
| Hidden Size | 768 |
| Num Layers | 12 |
| Vocab Size | 1880 (SAFE tokens) |
| Checkpoint V1 | checkpoints/model.ckpt |
| Checkpoint V2 | checkpoints/model_v2.ckpt |

#### Running examples (recommended path)

**De Novo Generation:**

```bash
cd /path/to/genmol

# Generate molecules using V1 model
python scripts/exps/denovo/run.py

# Generate molecules using V2 model
python scripts/exps/denovo/run.py -c scripts/exps/frag/hparams_v2.yaml
```

**Fragment-constrained Generation:**

```bash
# V1 model
python scripts/exps/frag/run.py

# V2 model (recommended for fragments)
python scripts/exps/frag/run.py -c scripts/exps/frag/hparams_v2.yaml
```

**Goal-directed Hit Generation (PMO Benchmark):**

```bash
# Generate molecules optimized for QED
python scripts/exps/pmo/run.py -o qed

# Generate molecules optimized for synthetic accessibility
python scripts/exps/pmo/run.py -o sa

# Generate molecules for GSK3b target
python scripts/exps/pmo/run.py -o gsk3b

# Generate molecules for JNK3 target
python scripts/exps/pmo/run.py -o jnk3
```

**Lead Optimization:**

```bash
# Optimize molecules with oracle and similarity threshold
python scripts/exps/lead/run.py -o ${oracle_name} -i ${start_mol_idx} -d ${sim_threshold}
```

**Python (optional):** Import and use the model programmatically - see scripts/exps/ for detailed API usage.

---

## Reference Resources

- **GitCode (primary)**: https://gitcode.com/AI4Science/GenMol
- **Official README**: https://gitcode.com/AI4Science/GenMol/blob/main/README.md
- **Additional reference**: https://github.com/NVIDIA-Digital-Bio/genmol
- **GenMol V1 Weights**: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/genmol_v1
- **GenMol V2 Weights**: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/genmol_v2
- **SAFE Representation**: https://github.com/NVIDIA-Digital-Bio/safe-mol
- **NVIDIA Open Model License**: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/