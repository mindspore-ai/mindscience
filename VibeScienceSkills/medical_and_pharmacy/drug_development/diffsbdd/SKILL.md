---
name: diffsbdd
description: diffsbdd (Diffusion for Structure-Based Drug Design) is an equivariant diffusion model for structure-based drug design. Use this model when you need to generate novel small molecule ligands that bind to specific protein binding pockets.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# DiffSBDD

## Overview

DiffSBDD (Diffusion for Structure-Based Drug Design) is an equivariant diffusion model that generates novel small molecule ligands for given protein binding pockets. Published in Nature Computational Science (2024), it uses a diffusion-based approach to create drug-like molecules conditioned on the 3D structure of protein binding sites.

The model supports three main tasks:
- **De novo design**: Generate new ligands from scratch for a given protein pocket
- **Substructure inpainting**: Design molecules around fixed substructures (scaffold elaboration, fragment linking)
- **Molecular optimization**: Optimize existing molecules for drug-likeness (QED) or synthetic accessibility (SA)

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits your task requirements.

- **Scenario 1**: De novo ligand design - Suitable for generating novel drug-like molecules that bind to a target protein pocket from scratch
- **Scenario 2**: Scaffold elaboration - Suitable for growing molecules around fixed fragments or substructures
- **Scenario 3**: Fragment linking - Suitable for connecting two or more fragments in a protein binding site
- **Scenario 4**: Molecular optimization - Suitable for improving drug-likeness or synthetic accessibility of existing molecules

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | PDB files (.pdb) containing protein structure with binding pocket |
| Reference Ligand | Optional SDF file or PDB residue specification (chain:residue) to define the pocket |
| Data Size | Single PDB files for inference; directories for batch testing |
| Data Source | RCSB PDB, custom experimental structures, or predicted structures |

#### Data Acquisition Methods

1. **RCSB PDB Download** - Download from https://www.rcsb.org/ - Search for protein-ligand complexes with known binding sites
2. **PDBBind Database** - Use curated protein-ligand complexes from http://www.pdbbind.org.cn/
3. **Custom PDB Files** - Prepare PDB files using molecular modeling software (PyMOL, Chimera)

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Ensure PDB file contains the protein structure with the binding pocket defined
- **Step 2**: Prepare reference ligand in SDF format or specify residue ID (chain:residue) in the PDB
- **Step 3**: For substructure inpainting, prepare a fixed fragment SDF file with atoms to preserve
- **Step 4**: Place PDB and SDF files in accessible directory paths for inference

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

**Note**: The README may **not** spell out HDK/CANN version numbers. Example `torch_npu` paths include **v6.0.rc3-pytorch2.1.0** and **2.1.0.post8** (cp311 aarch64). The conda example uses **Python 3.10.4** while some wheels are **cp311**; choose the wheel that matches your Python version.

| Component | Version (per README; NPU path) |
| --------- | ------------------------ |
| HDK       | Not listed in README; align with chosen CANN / torch_npu release notes |
| CANN      | Not listed in README; see Ascend PyTorch release pages and wheel tags |
| Python    | 3.10.4 |
| torch     | Pair with torch_npu 2.1.0.post8 (2.1.x family in README) |
| torch-npu | 2.1.0.post8 (README wget example) |

#### Clone repository (Ascend — primary)

```bash
git clone https://ai.gitcode.com/AI4Science/DiffSBDD.git
cd DiffSBDD
# Per README: conda python=3.10.4, pip dependencies, then download and install torch_npu wheel for NPU
```

#### Environment Requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend NPU (per GitCode README) |
| Memory | 8GB+ host RAM recommended |
| Disk Space | ~500MB for checkpoints; additional space for generated molecules |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | `git clone https://ai.gitcode.com/AI4Science/DiffSBDD.git` and create the environment per README |
| 2 | Install **torch_npu** (README wheel) and remaining pip dependencies |
| 3 | Download Zenodo weights into `checkpoints/` (README) |
| 4 | Run `python generate_ligands.py ...` (README examples) |

**Optional NPU check:**

```bash
python -c "import torch; import torch_npu; print(torch.__version__); print(torch_npu.npu.is_available())"
```

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Functional Limitations | Requires a reference ligand or residue list to define the binding pocket; cannot generate ligands without pocket definition |
| Performance Limitations | Throughput depends on device class and denoising settings |
| Scale Limitations | Large protein structures may require more memory; batch size limited by available device memory |
| Input Format | Requires clean PDB files; may fail with incomplete structures or missing atoms |

#### Notes

- **Note 1**: The model requires a reference ligand or residue list to define the binding pocket location
- **Note 2**: Generated molecules may need sanitization to remove invalid structures using the `--sanitize` flag
- **Note 3**: The `--relax` flag uses a force field that doesn't consider the protein and may introduce clashes
- **Note 4 (NPU)**: Use **https://ai.gitcode.com/AI4Science/DiffSBDD** README together with the selected **torch_npu** wheel

---

### 4. Model Invocation Guide

#### Model Initialization

| Item | Example / value |
| ---- | ---------------- |
| checkpoint | crossdocked_fullatom_cond.ckpt (recommended for general use) |
| Available models | crossdocked_ca_cond.ckpt, crossdocked_ca_joint.ckpt, crossdocked_fullatom_cond.ckpt, crossdocked_fullatom_joint.ckpt, moad_*.ckpt |

#### Running examples (recommended path)

**Download pre-trained models:**
```bash
mkdir -p checkpoints
wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt
```

**Shell - De novo ligand generation:**

```bash
cd /path/to/DiffSBDD

# Using reference ligand in PDB (chain:residue)
python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile example/3rfm.pdb \
    --outfile example/3rfm_mol.sdf \
    --ref_ligand A:330 \
    --n_samples 20

# Using reference ligand SDF file
python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile example/3rfm.pdb \
    --outfile example/3rfm_mol.sdf \
    --ref_ligand example/3rfm_B_CFF.sdf \
    --n_samples 20

# Using residue list (no reference ligand)
python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile 1abc.pdb \
    --outfile results/1abc_mols.sdf \
    --resi_list A:1 A:2 A:3 A:4 A:5 A:6 A:7 \
    --n_samples 20
```

**Shell - Substructure inpainting:**

```bash
python inpaint.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile example/5ndu.pdb \
    --outfile example/5ndu_linked_mols.sdf \
    --ref_ligand example/5ndu_C_8V2.sdf \
    --fix_atoms example/fragments.sdf \
    --center ligand \
    --add_n_nodes 10
```

**Shell - Molecular optimization:**

```bash
python optimize.py \
    --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile example/5ndu.pdb \
    --outfile output.sdf \
    --ref_ligand example/5ndu_C_8V2.sdf \
    --objective sa \
    --population_size 100 \
    --evolution_steps 10 \
    --top_k 10 \
    --timesteps 100
```

**Common optional flags:**

| Flag | Description |
|------|-------------|
| `--n_samples` | Number of sampled molecules (default: 10) |
| `--num_nodes_lig` | Size of sampled molecules |
| `--timesteps` | Number of denoising steps |
| `--all_frags` | Keep all disconnected fragments |
| `--sanitize` | Sanitize molecules (remove invalid) |
| `--relax` | Relax in force field |

#### Result Post-processing

- **Output format**: SDF file containing generated molecules
- **Visualization**: Use PyMOL, Chimera, or RDKit to visualize generated ligands in the protein pocket
- **Metrics**: Use RDKit to compute QED (Quantitative Estimate of Drug-likeness) and SA (Synthetic Accessibility) scores
- **Location**: Output files are written to the path specified by `--outfile`

---

## Reference Resources

- **GitCode (primary)**: https://ai.gitcode.com/AI4Science/DiffSBDD
- **Official README**: https://ai.gitcode.com/AI4Science/DiffSBDD/blob/main/README.md
- **Additional reference**: https://github.com/arneschneuing/DiffSBDD
- **Paper (Nature Computational Science)**: DOI: 10.1038/s43588-024-00737-x
- **ArXiv**: 2210.13695
- **Model Weights (Zenodo)**: https://zenodo.org/record/8183747