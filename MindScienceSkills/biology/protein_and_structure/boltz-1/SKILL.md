---
name: boltz-1
description: boltz-1 is a deep learning model for biomolecular interaction prediction. It was the first fully open-source model to approach AlphaFold3 accuracy for predicting 3D structures of protein-ligand, protein-protein, protein-DNA/RNA, and other biomolecular complexes. Use this model when you need to predict the structure and binding interactions of biomolecules.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# Boltz-1

## Overview

Boltz-1 is a deep learning model for biomolecular interaction prediction developed by jwohlwend et al. It represents the first fully open-source model to approach AlphaFold3 accuracy for predicting 3D structures of protein-ligand complexes, protein-protein interactions, protein-DNA/RNA complexes, and other biomolecular interactions. The model uses diffusion-based generative modeling to jointly predict complex structures and binding modes.

All model weights and code are provided under MIT license, making them freely available for both academic and commercial use. Boltz-1 supports various biomolecular complex types and provides predictions including structure coordinates, confidence scores, and optional binding affinity estimation.

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits their task requirements.

- **Scenario 1**: Protein-ligand structure prediction - Suitable for predicting the 3D binding pose of small molecule ligands to protein targets
- **Scenario 2**: Protein-protein complex prediction - Suitable for predicting quaternary structure and binding interfaces between protein chains
- **Scenario 3**: Protein-DNA/RNA interaction modeling - Suitable for predicting structures of nucleic acid-protein complexes
- **Scenario 4**: Binding affinity estimation - Suitable for predicting binding affinity values (log10(IC50) in μM) and binder probability

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | YAML files (.yaml) describing biomolecules to model |
| Input Types | PDB files (.pdb) for molecular structures, or MSA data |
| Data Size | Single YAML file or directory of YAML files for batched processing |
| Data Source | RCSB PDB, custom experimental structures, AlphaFold predictions |

#### Data Acquisition Methods

1. **RCSB PDB Download** - Download from https://www.rcsb.org/ - Search for biomolecular complexes of interest
2. **Custom PDB Files** - Prepare PDB files using molecular modeling software (PyMOL, Chimera)
3. **AlphaFold Predictions** - Use AlphaFold predictions as input for proteins without experimental structures

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Create a YAML input file describing the biomolecules to model (see prediction instructions in docs/prediction.md)
- **Step 2**: Ensure PDB files contain complete structure with all atoms
- **Step 3**: For ligand predictions, include ligand information in the YAML configuration
- **Step 4**: Place input YAML file(s) in an accessible directory path

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

**Note**: The official README may name the conda environment **Boltz-2** (documentation inconsistency possible). **torch_npu** wheel lines may list mixed **cp311 / cp310** artifacts; pick the wheel that matches your Python minor version.

| Component | Version |
| --------- | ------------------------------ |
| HDK       | 25.0.rc1.1 |
| CANN      | 8.1.RC1 |
| Python    | 3.11.6 |
| torch     | 2.3.1 |
| torch-npu | 2.3.1.post6 |

#### Clone / install (Ascend — summary)

```bash
git clone https://ai.gitcode.com/AI4Science/Boltz-1.git
cd Boltz-1
# Per README: conda, base pip deps, torch==2.3.1, install matching torch_npu wheel, install boltz extra per README
# Per README: patch boltz/main.py (import torch_npu, etc.) and Lightning Fabric accelerator hooks
```

#### Environment Requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend NPU (per GitCode README) |
| Memory | Large complexes benefit from ample device and host memory |
| Disk Space | ~2GB+ for weights and dependencies |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | `git clone https://ai.gitcode.com/AI4Science/Boltz-1.git` |
| 2 | Create conda env per README (example Python **3.11.6**) |
| 3 | Install **torch** / **torch_npu** and Boltz per README (including any patch steps) |
| 4 | Prepare input YAML |
| 5 | Run: `boltz predict input_path --use_msa_server` (or as specified in README) |
| 6 | Inspect output directory |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Hardware | Ascend deployment follows **https://ai.gitcode.com/AI4Science/Boltz-1** README |
| Performance | Throughput depends on device class, complex size, and batch settings |
| MSA Server | Requires `--use_msa_server` for automatic MSA generation; may need authentication for some servers |
| Input Format | Requires YAML configuration files; raw PDB files not directly supported |
| Complex Size | Very large complexes may require additional memory |

#### Notes

- **Note 1**: Boltz-1 uses the `--use_msa_server` flag to automatically generate multiple sequence alignments (MSAs) for improved predictions. This requires access to an MSA server.
- **Note 2**: For binding affinity prediction, two outputs are provided: `affinity_pred_value` (log10(IC50) in μM) for ligand optimization stages, and `affinity_probability_binary` (0-1 probability) for binder detection in hit discovery.
- **Note 3**: Optional accelerated kernels may be available depending on the installed package variant; see README.
- **Note 4**: A newer version, Boltz-2, is available and provides improved accuracy including binding affinity prediction approaching physics-based FEP methods.

---

### 4. Model Invocation Guide

#### Running inference (recommended path)

**Shell:**

```bash
# Activate environment
conda activate boltz_env

# Run prediction with MSA server (recommended)
boltz predict /path/to/input.yaml --use_msa_server

# For batch processing of multiple inputs
boltz predict /path/to/input_directory/ --use_msa_server

# View all available options
boltz predict --help
```

**Input YAML format example:**

```yaml
# Example: Protein-ligand complex
protein:
  id: 1ABC
  chain: A
  pdb: /path/to/protein.pdb
ligand:
  id: LIG
  pdb: /path/to/ligand.pdb
```

#### Result Post-processing

- **Output files**: Generated structure predictions in PDB format
- **Confidence scores**: Model provides confidence metrics for predicted structures
- **Affinity predictions** (if enabled): Binding affinity values and probability scores
- **Visualization**: Use PyMOL, Chimera, or other molecular visualization tools to inspect results

---

## Reference Resources

- **GitCode (primary)**: https://ai.gitcode.com/AI4Science/Boltz-1
- **Official README**: https://ai.gitcode.com/AI4Science/Boltz-1/blob/main/README.md
- **Additional reference**: https://github.com/jwohlwend/boltz
- **Boltz-1 Paper**: `https://doi.org/10.1101/2024.11.19.624167` - Technical report
- **Boltz-2 Paper**: `https://doi.org/10.1101/2025.06.14.659707` - Next version with improved affinity prediction
- **Prediction Documentation**: `https://github.com/jwohlwend/boltz/tree/main/docs/prediction.md` - Input format and options
- **ColabFold (MSA)**: `https://github.com/Mirdita/colabfold` - Referenced for MSA generation