---
name: esm3
description: esm3 (Evolutionary Scale Modeling 3) is a frontier generative model for biology developed by EvolutionaryScale. It can jointly reason across sequence, structure, and function to generate novel protein sequences, predict structures, and perform inverse folding. Use this model when you need to generate protein sequences, predict 3D protein structures, or perform structure-based sequence design.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# ESM3

## Overview

ESM3 (Evolutionary Scale Modeling 3) is a frontier generative model for biology developed by EvolutionaryScale. It represents a paradigm shift in computational biology by treating proteins as a multi-track generative problem, jointly reasoning across three fundamental biological properties:

- **Sequence** (amino acid sequences)
- **Structure** (3D protein structure)  
- **Function (function annotations)

The model uses a generative masked language model architecture that can be prompted with partial information from any of these three tracks and iteratively sample masked positions. This enables diverse applications including novel protein design, inverse folding, structure prediction, and function annotation.

**Available Models:**

| Model | Parameters | Description |
|-------|------------|-------------|
| esm3-large-2024-03 | 98B | Largest model, highest capability |
| esm3-medium-2024-08 | 7B | Medium-sized model |
| esm3-small-2024-08 | 1.4B | Small model |
| esm3-open | 1.4B | Open weights version (esm3-small) |

**ESM C (Cambrian)** - Parallel representation learning models:

| Model | Parameters |
|-------|------------|
| esmc-6b-2024-12 | 6B |
| esmc-600m-2024-12 | 600M |
| esmc-300m-2024-12 | 300M |

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits your task requirements.

- **Scenario 1**: Protein sequence generation - Suitable for generating novel protein sequences from scratch or continuing from partial sequences
- **Scenario 2**: Protein structure prediction - Suitable for predicting 3D protein structures from sequences (via structure track generation)
- **Scenario 3**: Inverse folding / sequence design - Suitable for designing protein sequences that fold to a given structure
- **Scenario 4**: Function-conditioned generation - Suitable for generating proteins with specific functional annotations

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | ESMProtein objects (Python), PDB files, or raw amino acid sequences |
| Data Size | Single sequences or batches; length limited by model context (typically up to ~2000 residues) |
| Data Source | Custom sequences, RCSB PDB, or AlphaFold predictions |

#### Data Acquisition Methods

1. **Direct Sequence Input** - Provide amino acid sequences as strings (use `_` for masked positions)
2. **PDB File Loading** - Load existing protein structures using `ESMProtein.from_pdb("file.pdb")`
3. **HuggingFace Hub** - Access pre-trained model weights from EvolutionaryScale/esm3

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Install ESM package: `pip install esm`
- **Step 2**: Login to HuggingFace (for model download): `huggingface-cli login` or `from huggingface_hub import login; login()`
- **Step 3**: Prepare input as ESMProtein object with sequence/structure/function data
- **Step 4**: Use underscores (`_`) to mask positions for generation

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

**Note**: The component table lists **Python 3.10**, while some conda examples use **python=3.11**. Use the Python minor version you actually run and keep **torch_npu** matched to CANN per README.

| Component | Version |
| --------- | ------------------------------ |
| HDK       | 24.1.RC3 |
| CANN      | 8.2.RC1 |
| Python    | 3.10 |
| torch     | 2.6.0 |
| torch-npu | 2.6.0 |

#### Clone / install (Ascend — summary)

```bash
git clone https://ai.gitcode.com/AI4Science/esm3.git
cd esm3
# Per README: conda env, cd esm, pip install esm and listed deps, pip install torch_npu==2.6.0
```

#### Conda environment and Python dependencies

```bash
conda create --name esm3_env python=3.11
conda activate esm3_env
cd esm
pip install esm
pip install cloudpickle ml-dtypes psutil tornado absl-py httpx numpy==1.26.3
pip install torch_npu==2.6.0
# Weights and inference: see README sections on download and inference
```

**Dependencies (reference list; exact pins follow the official README and `pyproject.toml`):**
- `torch>=2.2.0`
- `transformers<4.53.0`
- `einops`
- `biotite>=1.0.0`
- `biopython`
- `scikit-learn`
- `pandas`
- `httpx`
- `boto3`
- `py3dmol`
- `pydssp`

#### Download model parameters

Model weights are automatically downloaded from HuggingFace Hub when calling `from_pretrained()`. No manual download required.

```python
from esm.models.esm3 import ESM3
model = ESM3.from_pretrained("esm3-open")  # Downloads automatically
```

**Available model IDs on HuggingFace:**
- `esm3-open` / `esm3_sm_open_v1` - Small open model (1.4B)
- `esm3-medium-2024-08` - Medium model (7B)
- `esm3-large-2024-03` - Large model (98B)
- `esmc_300m` - ESM C 300M
- `esmc_600m` - ESM C 600M
- `esmc_6b_2024_12` - ESM C 6B

#### Environment Requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend NPU (per GitCode / ai.gitcode README) |
| Memory | Small models need modest device memory; large checkpoints (e.g. 98B scale) need very large memory (see README) |
| Disk Space | ~3GB for model weights (esm3-open), ~20GB+ for larger models |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | Create conda environment per README (e.g. Python 3.11) |
| 2 | Install ESM and `torch_npu` per README |
| 3 | Login to HuggingFace (if accessing gated models): `huggingface-cli login` |
| 4 | Load model: `model = ESM3.from_pretrained("esm3-open").to("npu")` (device string per README) |
| 5 | Prepare input ESMProtein with sequence/structure |
| 6 | Run generation: `model.generate(protein, GenerationConfig(...))` |
| 7 | Save output: `protein.to_pdb("output.pdb")` |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Functional Limitations | Cannot generate function annotations from scratch without prompting; requires at least one track (sequence/structure/function) as input |
| Performance Limitations | Large models (98B) require very large device memory; inference speed depends on sequence length and generation steps |
| Scale Limitations | Maximum sequence length limited by model context (~2000 residues); large models may OOM on longer sequences |
| Input Format | Must use ESMProtein object; raw strings not directly accepted for generation |

#### Notes

- **Note 1**: ESM3 uses a multi-track generation approach. You can generate sequence, structure, or function tracks independently or jointly by specifying the `track` parameter in `GenerationConfig`.
- **Note 2 (runtime)**: Supported devices and install steps are defined by the repository README you follow (GitCode Ascend adaptation vs. other distributions).
- **Note 3**: Use underscores (`_`) in the sequence to mask positions that should be generated. The model will fill in these positions.
- **Note 4**: For best results, use `num_steps=8` or higher for sequence generation and `num_steps=8` for structure generation.
- **Note 5**: ESM C models (esmc_300m, esmc_600m, esmc_6b) are designed for representation learning and embeddings, not generation. Use ESM3 for generative tasks.

---

### 4. Model Invocation Guide

#### Model Initialization

| Item | Example / value |
| ---- | ---------------- |
| Model ID (open) | esm3-open |
| Model ID (small) | esm3-small-2024-08 |
| Model ID (medium) | esm3-medium-2024-08 |
| Model ID (large) | esm3-large-2024-03 |
| Device | Per README (e.g. `"npu"` for Ascend path) |

#### Running examples (recommended path)

**Python - Generate protein sequence from masked input:**

```python
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Login to HuggingFace (for model download)
login(token="<your-huggingface-token>")

# Load model
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("npu")

# Create protein with partial sequence (underscores are masked positions)
prompt = "M" * 50 + "_" * 100 + "M" * 50  # 200 aa total, 100 masked
protein = ESMProtein(sequence=prompt)

# Generate sequence
protein = model.generate(protein, GenerationConfig(
    track="sequence", 
    num_steps=8, 
    temperature=0.7
))

# Generate structure
protein = model.generate(protein, GenerationConfig(
    track="structure", 
    num_steps=8
))

# Save to PDB
protein.to_pdb("./generation.pdb")
print(f"Generated sequence: {protein.sequence}")
```

**Python - Load from PDB and predict:**

```python
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

model = ESM3.from_pretrained("esm3-open").to("npu")

# Load existing structure
protein = ESMProtein.from_pdb("input.pdb")

# Generate sequence for the structure (inverse folding)
protein = model.generate(protein, GenerationConfig(
    track="sequence", 
    num_steps=8
))

protein.to_pdb("designed.pdb")
```

**Python - Using ESM C for embeddings:**

```python
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

# ESM C is for representation learning, not generation
client = ESMC.from_pretrained("esmc_300m").to("npu")

protein = ESMProtein(sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH")
protein_tensor = client.encode(protein)

logits_output = client.logits(
    protein_tensor, 
    LogitsConfig(sequence=True, return_embeddings=True)
)

print(f"Embeddings shape: {logits_output.embeddings.shape}")
```

#### Result Post-processing

- **Output format**: PDB files via `protein.to_pdb("output.pdb")`
- **Sequence**: Access via `protein.sequence`
- **Structure**: Stored in `protein.coordinates` (atom37 format)
- **Function**: Stored in `protein.function_annotations`

---

## Reference Resources

- **GitCode (primary)**: https://ai.gitcode.com/AI4Science/esm3
- **Official README**: https://ai.gitcode.com/AI4Science/esm3/blob/main/README.md
- **Additional reference**: https://github.com/evolutionaryscale/esm/tree/main
- **HuggingFace Model Hub**: `https://huggingface.co/EvolutionaryScale/esm3`
- **PyPI Package**: `https://pypi.org/project/esm/`