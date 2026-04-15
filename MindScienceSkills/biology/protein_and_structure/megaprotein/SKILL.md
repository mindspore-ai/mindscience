---
name: megaprotein
description: megaprotein is a comprehensive protein structure prediction toolkit developed by MindSpore scientific computing team. It consists of three components:MEGA-Fold (protein structure prediction), MEGA-EvoGen (MSA generation), and MEGA-Assessment (structure quality assessment). Use this model when you need to predict 3D protein structures, generate multiple sequence alignments, or evaluate protein structure quality on Ascend NPU.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend

---

# MEGAProtein

## Overview

MEGA-Protein is a protein structure prediction toolkit developed by the MindSpore scientific computing team in collaboration with Professor Gao Yiqin's team. It addresses limitations of traditional structure prediction tools and AlphaFold2, including slow data preprocessing, poor accuracy without MSA, and lack of universal structure quality evaluation tools.

The toolkit consists of three main components:

- **MEGA-Fold**: Protein structure prediction tool with network architecture similar to AlphaFold2, using MMseqs2 for sequence search (2-3x faster than original). Achieved CAMEO-3D monthly leaderboard #1 in April 2022.
- **MEGA-EvoGen**: MSA generation tool that improves single-sequence prediction speed and maintains accuracy even with few-shot or zero-shot MSA scenarios. Achieved CAMEO-3D monthly leaderboard #1 in July 2022.
- **MEGA-Assessment**: Protein structure scoring tool that evaluates per-residue accuracy and residue-residue distance errors. Achieved CAMEO-QE monthly leaderboard #1 in July 2022.

This skill provides inference capabilities adapted for Ascend NPU, enabling users to run protein structure prediction tasks on Huawei Ascend NPUs.

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

- **Protein structure prediction**: Predict 3D protein structures from amino acid sequences
- **MSA generation**: Generate multiple sequence alignments for proteins with limited or no MSA data (orphan sequences, highly variable sequences, designed proteins)
- **Structure quality assessment**: Evaluate the accuracy of predicted protein structures
- **Drug discovery**: Identify protein targets and evaluate binding site structures
- **Protein engineering**: Analyze designed proteins or enzyme structures

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | FASTA sequences (.fasta), PDB files (.pdb), or pickle feature files |
| Data Size   | Single sequences or directories with multiple sequences     |
| Data Source | Input sequences from users, RCSB PDB, AlphaFold DB, or custom sequences |

#### Data Acquisition Methods

1. **Direct sequence input**: Provide amino acid sequences in FASTA format
2. **MSA pickle files**: Use pre-computed MSA features in pickle format
3. **PDB files**: Load protein structures from PDB format files

#### Data Preprocessing

- Ensure sequences contain only standard amino acid letters (ACDEFGHIKLMNPQRSTVWY)
- For MEGA-EvoGen: Can work with single sequences (zero-shot) or with existing MSA (few-shot)
- For MEGA-Fold: Requires MSA features from MEGA-EvoGen or traditional database search
- For MEGA-Assessment: Requires both MSA features and predicted PDB structure

---

### 2. Environment Configuration and Dependencies

#### Verified Ascend stack (reference)

| Component | Version              |
| --------- | -------------------- |
| CANN      | 8.3.RC1.alpha001     |
| Python    | 3.8+ (recommended)  |
| mindspore | 2.7.1.post1          |
| rdkit     | 2024.3.5             |

Install **CANN** on the host per Huawei documentation before the Python steps.

#### Clone MindSpore Science repository and prepare environment

From a working directory of your choice:

```bash
git clone -b legacy-master https://gitee.com/mindspore/mindscience.git
cd mindscience/MindSponge
pip install -r requirements.txt
export PYTHONPATH=mindscience/MindSPONGE/src:${PYTHONPATH}
```

#### Environment Requirements (hardware / disk)

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Hardware    | Ascend 910 (32GB memory recommended)          |
| Memory      | 32GB for sequences up to 2048 residues        |
| Disk Space  | Varies based on model weights and datasets    |

#### Installation Steps

1. **Install CANN**: Install CANN 8.3.RC1.alpha001 on the Ascend host per Huawei documentation
2. **Clone repository**: Clone the MindSpore mindscience repository
3. **Install dependencies**: Run `pip install -r requirements.txt`
4. **Set PYTHONPATH**: Export `PYTHONPATH=mindscience/MindSPONGE/src:${PYTHONPATH}`
5. **Verify installation**: Run a simple inference test to verify setup

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| ----------------| ------------|
| Functional Limitations | MEGA-Assessment requires both MSA features and PDB structure as input |
| Performance Limitations | Sequence length limited by available memory (2048 for 32GB, 3072 in Pipeline mode) |
| Scale Limitations | MEGA-EvoGen supports max 768 residues on 32GB Ascend 910 |

#### Notes

- **Note 1**: MEGA-EvoGen can generate MSA features from single sequences (zero-shot) or enhance existing MSA (few-shot)
- **Note 2**: For traditional database MSA search, refer to `application/common_utils/database_query/README.md`
- **Note 3**: Post-processing with Amber force field is recommended to add hydrogen atoms and optimize structure (refer to `application/common_utils/openmm_relaxation/README.md`)
- **Note 4**: Training and inference weights require conversion using `get_predict_checkpoint` and `get_train_checkpoint` utilities

---

### 4. Model Invocation Guide

#### MEGA-EvoGen (MSA Generation)

```python
import numpy as np
import mindspore as ms
from mindsponge import PipeLine
from mindsponge.common.config_load import load_config

ms.set_context(mode=ms.GRAPH_MODE)

# Input sequence
fasta = "GYDKDLCEWSMTADQTEVETQIEADIMNIVKRDRPEMKAEVQKQLKSGGVMQYNYVLYCDKNFNNKNIIAEVVGE"

# Initialize MEGA-EvoGen
msa_generator = PipeLine(name="MEGAEvoGen")
msa_generator.set_device_id(0)
local_config_path = "./model_configs/MEGAEvoGen/evogen_predict_256.yaml"
conf = load_config(local_config_path)
msa_generator.initialize(conf=conf)
msa_generator.model.from_pretrained()

# Generate MSA features
msa_feature = msa_generator.predict(fasta)
```

#### MEGA-Fold (Structure Prediction)

```python
import pickle
import mindspore as ms
from mindsponge import PipeLine
from mindsponge.common.config_load import load_config
import os
import stat

ms.set_context(mode=ms.GRAPH_MODE)

# Load pre-computed MSA features
with open("./test.pkl", "rb") as f:
    feature = pickle.load(f)

# Initialize MEGA-Fold
fold_prediction = PipeLine(name="MEGAFold")
local_config_path = "./model_configs/MEGAFold/predict_256.yaml"
conf = load_config(local_config_path)

fold_prediction.set_device_id(0)
fold_prediction.initialize(conf=conf)
fold_prediction.model.from_pretrained()

# Predict structure
res = fold_prediction.predict(feature)
pdb_file = res[-1]

# Save PDB file
pdb_path = './res.pdb'
os_flags = os.O_RDWR | os.O_CREAT
os_modes = stat.S_IRWXU
with os.fdopen(os.open(pdb_path, os_flags, os_modes), 'w') as fout:
    fout.write(pdb_file)
```

#### MEGA-Assessment (Structure Quality Assessment)

```python
import pickle
import numpy as np
from mindspore import context
from mindsponge import PipeLine
from mindsponge.common.config_load import load_config
from mindsponge.common.protein import from_pdb_string

protein_assessment = PipeLine(name="MEGAAssessment")
protein_assessment.set_device_id(0)

local_config_path = "./mindscience/MindSPONGE/applications/model_configs/MEGAFold/predict_256.yaml"
conf = load_config(local_config_path)

protein_assessment.initialize(conf=conf)
protein_assessment.model.from_pretrained()

# Load raw MSA feature
with open("./test.pkl", "rb") as f:
    raw_feature = pickle.load(f)

# Load predicted PDB structure
with open('./res.pdb', 'r') as f:
    decoy_prot_pdb = from_pdb_string(f.read())

raw_feature['decoy_aatype'] = decoy_prot_pdb.aatype
raw_feature['decoy_atom_positions'] = decoy_prot_pdb.atom_positions
raw_feature['decoy_atom_mask'] = decoy_prot_pdb.atom_mask

# Evaluate structure
res = protein_assessment.predict(raw_feature)
print("score is:", np.mean(res))
```

#### Complete Pipeline (MEGA-EvoGen → MEGA-Fold → MEGA-Assessment)

```python
import numpy as np
import mindspore as ms
from mindsponge import PipeLine
from mindsponge.common.config_load import load_config

ms.set_context(mode=ms.GRAPH_MODE)

# Step 1: MEGA-EvoGen - Generate MSA features
fasta = "GYDKDLCEWSMTADQTEVETQIEADIMNIVKRDRPEMKAEVQKQLKSGGVMQYNYVLYCDKNFNNKNIIAEVVGE"
msa_generator = PipeLine(name="MEGAEvoGen")
msa_generator.set_device_id(0)
local_config_path = "./model_configs/MEGAEvoGen/evogen_predict_256.yaml"
conf = load_config(local_config_path)
msa_generator.initialize(conf=conf)
msa_generator.model.from_pretrained()
msa_feature = msa_generator.predict(fasta)

# Step 2: MEGA-Fold - Predict protein structure
fold_prediction = PipeLine(name="MEGAFold")
fold_prediction.set_device_id(0)
local_config_path = "./model_configs/MEGAFold/predict_256.yaml"
conf = load_config(local_config_path)
fold_prediction.initialize(conf=conf)
fold_prediction.model.from_pretrained()
final_atom_positions, final_atom_mask, aatype, _, _ = fold_prediction.model.predict(msa_feature)

# Step 3: MEGA-Assessment - Evaluate structure quality
protein_assessment = PipeLine(name="MEGAAssessment")
protein_assessment.set_device_id(0)
protein_assessment.initialize(conf=conf)
protein_assessment.model.from_pretrained()
msa_feature['decoy_aatype'] = np.pad(aatype, (0, 256 - aatype.shape[0]))
msa_feature['decoy_atom_positions'] = np.pad(final_atom_positions, ((0, 256 - final_atom_positions.shape[0]), (0, 0), (0, 0)))
msa_feature['decoy_atom_mask'] = np.pad(final_atom_mask, ((0, 256 - final_atom_mask.shape[0]), (0, 0)))

res = protein_assessment.model.predict(msa_feature)
print("score is:", np.mean(res[:msa_feature['num_residues']]))
```

---

## Reference Resources

- **MindSpore MindSponge Official Repository**: https://gitee.com/mindspore/mindscience
- **MEGA-Fold Paper**: https://arxiv.org/abs/2206.12240
- **MEGA-EvoGen Paper**: https://doi.org/10.1021/acs.jctc.3c00528
- **PSP Dataset**: http://ftp.cbi.pku.edu.cn/psp/
- **CAMEO Benchmark**: https://www.cameo3d.org/
- **AlphaFold2 Paper**: https://www.nature.com/articles/s41586-021-03819-2
- **MMseqs2 Paper**: https://www.biorxiv.org/content/10.1101/2021.08.15.456425v1.full.pdf