---
name: proteinmpnn
description: ProteinMPNN is a deep learning method for protein sequence design given a protein backbone structure. Use this model when you need to design amino acid sequences that fold into target 3D structures, including monomer design, complex design, and antibody CDR design.
license: MIT License (original), Apache License 2.0 (MindSpore implementation)
metadata:
    skill-author: MindSpore Science Team
---

# ProteinMPNN

**IMPORTANT:** This SKILL.md should only document **inference-related** functionality. Do not include training, fine-tuning, evaluation, or other non-inference scenarios. All sections should focus on how to use the model for inference/prediction tasks only.

## Overview

ProteinMPNN is a deep learning method for protein sequence design that generates high-quality amino acid sequences capable of folding into target 3D structures given a protein backbone. This implementation is based on MindSpore and adapted from the original [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) repository, with integration for [RFantibody](https://github.com/RosettaCommons/RFantibody) antibody design tools.

Key capabilities:
- Fast sequence design for protein backbones
- Support for monomer and complex design
- Antibody CDR loop design
- Flexible position fixing and tied positions for symmetric design
- PSSM-guided design

---

## When to Use

- **Monomer Design**: Design sequences for single-chain protein structures
- **Complex Design**: Design sequences for multi-chain protein complexes with optional fixed chains
- **Antibody CDR Design**: Design CDR loops for antibody sequences in HLT format
- **Score-Only Mode**: Evaluate existing backbone-sequence pairs without generating new sequences
- **Guided Design (optional)**: Use PSSM or amino acid bias to guide sequence design
- **Symmetric Design (optional)**: Design homooligomers with tied positions across chains

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | PDB files (standard protein structure format), JSONL (parsed chains), FASTA (for score-only mode) |
| Data Size | Single PDB to thousands of structures; batch processing supported |
| Data Source | Experimental structures (PDB database), predicted structures (AlphaFold, RoseTTAFold), or designed structures |

#### Data Acquisition Methods

1. **PDB Database**: Download structures from RCSB PDB (https://www.rcsb.org/) - Standard source for experimental structures
2. **Structure Prediction**: Use AlphaFold, RoseTTAFold, or similar tools - For predicted structures
3. **Example Data**: Use provided example inputs - Quick testing and validation

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Parse PDB files**: Convert PDB files to JSONL format using `parse_multiple_chains.py` helper script
- **Assign chains (optional)**: Specify which chains to design vs. fix using `assign_fixed_chains.py`
- **Fixed positions (optional)**: Create position dictionaries for partial design using `make_fixed_positions_dict.py`
- **Tied positions (optional)**: Create tied position dictionaries for symmetric design using `make_tied_positions_dict.py`

---

### 2. Environment Configuration and Dependencies

#### Dependency Installation

```bash
# Clone the repository
git clone https://gitee.com/mindspore/mindscience.git
cd mindscience/MindSPONGE/applications/proteinmpnn

# Download model weights
bash scripts/download_weights.sh

# Install dependencies
pip install -r requirements.txt

# Extract example PDBs for testing
unzip examples/example_inputs.zip -d examples/
```

#### Environment Requirements

| Requirement | Specification |
|-------------|---------------|
| Python Version | Python >= 3.11 |
| Framework | MindSpore >= 2.7.1 |
| Hardware | Ascend NPU (CANN >= 8.2.RC1) or GPU |
| Memory | Sufficient for batch processing (adjust batch_size as needed) |
| Disk Space | ~1GB for model weights and examples |

#### Installation Steps

1. **Step 1**: Clone repository - `git clone https://gitee.com/mindspore/mindscience.git`
2. **Step 2**: Download weights - Run `bash scripts/download_weights.sh` to download pretrained model weights
3. **Step 3**: Install dependencies - Run `pip install -r requirements.txt`
4. **Step 4**: Verify installation - Run example scripts to verify the environment

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
|----------------|-------------|
| Functional Limitations | CA-only soluble model not yet available; requires backbone structure as input |
| Performance Limitations | Large complexes may require reduced batch size; sampling temperature affects diversity vs. quality tradeoff |
| Scale Limitations | Maximum sequence length configurable via --max_length (default 200000) |

#### Notes

- **Temperature Selection**: Lower temperatures (0.1-0.3) recommended for higher quality; higher values increase diversity but may reduce foldability
- **Model Variants**: Four versions available (v_48_002, v_48_010, v_48_020, v_48_030) trained with different noise levels
- **Chain Selection**: All chains designed by default; use chain_id_jsonl to specify fixed chains
- **Memory Management**: Reduce batch_size if encountering memory issues

#### License Agreement

This model uses MIT License (original) and Apache License 2.0 (MindSpore implementation). The main terms include:

- Free for academic and commercial use
- Attribution required for original ProteinMPNN paper
- See LICENSE file for complete terms

---

### 4. Model Invocation Guide

#### Invocation Process Overview

The basic model invocation process includes the following key steps:

1. **Model Loading**: Load pretrained ProteinMPNN model weights
2. **Data Preparation**: Parse PDB files and prepare input dictionaries
3. **Inference Execution**: Run sequence design with specified parameters
4. **Result Processing**: Output designed sequences in FASTA/PDB format

#### Complete Invocation Example

##### Example 1: Monomer Design

```bash
#!/bin/bash
folder_with_pdbs="path/to/pdbs/"
output_dir="output/monomer_design"

mkdir -p $output_dir
path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"

# Parse PDB files
python helper_scripts/parse_multiple_chains.py \
    --input_path=$folder_with_pdbs \
    --output_path=$path_for_parsed_chains

# Run sequence design
python proteinmpnn_run.py \
    --jsonl_path $path_for_parsed_chains \
    --out_folder $output_dir \
    --num_seq_per_target 2 \
    --sampling_temp "0.1" \
    --seed 37 \
    --batch_size 1
```

##### Example 2: Complex Design with Fixed Chains

```bash
#!/bin/bash
folder_with_pdbs="path/to/complexes/"
output_dir="output/complex_design"
chains_to_design="A B"  # Design chains A and B, fix others

mkdir -p $output_dir
path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
path_for_assigned_chains=$output_dir"/assigned_pdbs.jsonl"

# Parse PDB files
python helper_scripts/parse_multiple_chains.py \
    --input_path=$folder_with_pdbs \
    --output_path=$path_for_parsed_chains

# Assign which chains to design
python helper_scripts/assign_fixed_chains.py \
    --input_path=$path_for_parsed_chains \
    --output_path=$path_for_assigned_chains \
    --chain_list "$chains_to_design"

# Run sequence design
python proteinmpnn_run.py \
    --jsonl_path $path_for_parsed_chains \
    --chain_id_jsonl $path_for_assigned_chains \
    --out_folder $output_dir \
    --num_seq_per_target 2 \
    --sampling_temp "0.1" \
    --seed 37 \
    --batch_size 1
```

##### Example 3: Single PDB Design

```bash
#!/bin/bash
path_to_PDB="path/to/structure.pdb"
output_dir="output/single_pdb"
chains_to_design="A B"

python proteinmpnn_run.py \
    --pdb_path $path_to_PDB \
    --pdb_path_chains "$chains_to_design" \
    --out_folder $output_dir \
    --num_seq_per_target 2 \
    --sampling_temp "0.1" \
    --seed 37 \
    --batch_size 1
```

##### Example 4: Antibody CDR Design

```bash
python proteinmpnn_interface_design.py \
    -pdbdir /path/to/inputdir \
    -outpdbdir /path/to/outputdir \
    -loop_string "H1,H2,H3,L1,L2,L3" \
    -seqs_per_struct 1 \
    -temperature 0.000001
```

##### Example 5: Score-Only Mode

```bash
# Score existing backbone-sequence pairs
python proteinmpnn_run.py \
    --pdb_path "path/to/structure.pdb" \
    --out_folder "output/scores" \
    --score_only 1 \
    --save_score 1
```

#### Key Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_seq_per_target` | 1 | Number of sequences to generate per target |
| `--sampling_temp` | "0.1" | Sampling temperature (0.1-0.3 recommended) |
| `--batch_size` | 1 | Batch size (reduce for memory constraints) |
| `--seed` | 0 | Random seed (0 = random) |
| `--model_name` | v_48_020 | Model variant (v_48_002, v_48_010, v_48_020, v_48_030) |
| `--use_soluble_model` | False | Use soluble-only trained model |

#### Helper Scripts

The `scripts/` directory contains helper scripts for simplified ProteinMPNN usage:

- **`scripts/proteinmpnn_design.py`**: A unified interface for running ProteinMPNN design tasks

```python
# Example usage of helper script
from scripts.proteinmpnn_design import run_proteinmpnn, parse_multiple_chains

# Parse PDB files
parse_multiple_chains(
    input_path="path/to/pdbs/",
    output_path="parsed_pdbs.jsonl"
)

# Run sequence design
run_proteinmpnn(
    pdb_path="path/to/structure.pdb",
    output_dir="output/",
    chains="A B",
    num_seq_per_target=2,
    sampling_temp="0.1"
)
```

For command-line usage:
```bash
# Single PDB design
python scripts/proteinmpnn_design.py --pdb_path structure.pdb --output_dir output/ --chains "A B"

# Batch design from directory
python scripts/proteinmpnn_design.py --pdb_dir pdbs/ --output_dir output/ --num_seq 5
```

---

## Reference Resources

### Official Documentation

- [ProteinMPNN Paper](https://www.science.org/doi/10.1126/science.add2187): Original research paper in Science
- [Original GitHub Repository](https://github.com/dauparas/ProteinMPNN): Original PyTorch implementation
- [MindSPONGE Repository](https://gitee.com/mindspore/mindscience): MindSpore implementation

### Related Tutorials

- [MindSPONGE Documentation](https://www.mindspore.cn/mindsponge/docs): Official MindSPONGE documentation
- [RFantibody Integration](https://github.com/RosettaCommons/RFantibody): Antibody design tools

### Community Support

- [MindSpore Forum](https://bbs.huaweicloud.com/forum/forum-1076-1.html): Community discussions
- [Gitee Issues](https://gitee.com/mindspore/mindscience/issues): Bug reports and feature requests