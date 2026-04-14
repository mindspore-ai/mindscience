---
name: oligoformer
description: oligoformer is a transformer-based deep learning model for siRNA efficacy prediction. Use this model when you need to predict the effectiveness of small interfering RNA (siRNA) molecules for gene silencing through RNA interference (RNAi).
license: Non-commercial (Academic/Research)
metadata:
    skill-author: MindSpore Science Team
---

# OligoFormer

## Overview

OligoFormer is a transformer-based deep learning model designed for **siRNA (small interfering RNA) efficacy prediction**. It predicts the effectiveness of siRNA molecules for gene silencing through RNA interference (RNAi) by analyzing mRNA and siRNA sequence interactions. The model leverages transformer architecture to capture multi-dimensional sequence features and includes additional modules for off-target prediction and toxicity prediction.

The model was developed by Tsinghua University and supports both inter-dataset and intra-dataset training/testing scenarios. It uses RNA-FM (RNA Foundation Model) for feature extraction from RNA sequences.

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits their task requirements.

- **Scenario 1**: siRNA efficacy prediction - Suitable for predicting the effectiveness of siRNA molecules for gene silencing based on mRNA and siRNA sequence interactions
- **Scenario 2**: Off-target prediction - Suitable for identifying potential off-target effects of siRNA sequences
- **Scenario 3**: siRNA toxicity prediction - Suitable for assessing potential toxicity of siRNA molecules

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | FASTA files (.fa) for mRNA and siRNA sequences; CSV files for training |
| Data Size | Single sequences or batch processing; training typically uses datasets with 2,000+ siRNA samples |
| Data Source | Pre-trained model available; training data from published siRNA datasets (Huesken, Reynolds, Vickers, etc.) |

#### Data Acquisition Methods

1. **Pre-trained Model Download** - Download `best_model.pth` and `mismatch_model.pth` from the GitHub releases or Google Drive
2. **RNA-FM Feature Extractor** - Download RNA-FM pre-trained models for sequence feature extraction
3. **Training Data** - Use published siRNA datasets (Hu.csv, Mix.csv, Taka.csv) for custom training

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Prepare input mRNA sequence in FASTA format
- **Step 2** (optional): Prepare specific siRNA sequences in FASTA format if known
- **Step 3**: Ensure sequences use standard nucleotide letters (A, T, G, C, U)
- **Step 4**: For training, prepare CSV files with siRNA sequence and efficacy labels

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

| Component | Version |
| --------- | ------------------------------ |
| HDK       | 24.1.RC3 |
| CANN      | 8.2.RC1 |
| Python    | 3.10 |
| torch     | 2.6.0 |
| torch-npu | 2.6.0 |

#### Clone repository (GitCode — primary)

```bash
git clone https://gitcode.com/AI4Science/OligoFormer.git
cd OligoFormer
# Mirror in README: https://atomgit.com/AI4Science/OligoFormer.git
```

#### Conda environment and Python dependencies (README summary)

```bash
conda create --name OligoFormer python=3.10
conda activate OligoFormer
pip install -r requirements.txt
pip install torch_npu==2.6.0
# RNA-FM, tcmalloc, TASK_QUEUE, etc.: see README
```

#### Download RNA-FM (feature extraction)

```bash
# Option 1: Download packaged RNA-FM
wget https://cloud.tsinghua.edu.cn/f/46d71884ee8848b3a958/?dl=1 -O RNA-FM.tar.gz
tar -zxvf RNA-FM.tar.gz

# Option 2: Clone RNA-FM repository
git clone https://github.com/ml4bio/RNA-FM.git
cd RNA-FM
conda env create --name RNA-FM -f environment.yml

# Download pre-trained models from Google Drive
# https://drive.google.com/drive/folders/1VGye74GnNXbUMKx6QYYectZrY7G2pQ_J
```

#### Download model parameters

```bash
# Download best_model.pth and mismatch_model.pth from GitHub releases or Google Drive
# Place in OligoFormer/models/ directory
```

#### Environment Requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend NPU (per GitCode README) |
| Memory | 16GB+ RAM recommended |
| Disk Space | ~500MB for model parameters and RNA-FM, ~200MB for code |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | Clone: `git clone https://gitcode.com/AI4Science/OligoFormer.git` |
| 2 | Create conda environment per README; install `requirements.txt` and `torch_npu` |
| 3 | Download RNA-FM package and extract; download pre-trained RNA-FM models |
| 4 | Download OligoFormer model files (best_model.pth, mismatch_model.pth) |
| 5 | Prepare input FASTA file with mRNA sequence |
| 6 | Run inference: `python scripts/main.py --infer 1 -i1 data/example.fa` |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Functional Limitations | Requires RNA-FM for feature extraction |
| Performance Limitations | Throughput depends on device class and batch settings |
| Scale Limitations | Single sequence or batch processing; no streaming support |
| Input Format | Requires FASTA format with standard nucleotide letters (A, T, G, C, U) |
| License Restrictions | Non-commercial use only; commercial use requires authorization from Tsinghua University |

#### Notes

- **Note 1**: OligoFormer supports three inference modes:
  - Mode 1: Input FASTA file (traverses mRNA with 19nt sliding window)
  - Mode 2: Input specific mRNA and siRNA FASTA files
  - Mode 3: Manual mRNA sequence input
- **Note 2**: Off-target prediction requires the `mismatch_model.pth` file
- **Note 3**: The model uses a 19-nucleotide window for siRNA target site scanning
- **Note 4**: Docker image available for easier deployment: `yilanbai/oligoformer:v1.0`

---

### 4. Model Invocation Guide

#### Model Initialization

| Item | Example / value |
| ---- | ---------------- |
| Main model | best_model.pth (~5.8MB) |
| Mismatch model | mismatch_model.pth (~78KB) |
| Default inference mode | --infer 1 (FASTA input) |

#### Running examples (recommended path)

**Shell:**

```bash
cd /path/to/OligoFormer

# Inference Mode 1: Input FASTA file (traverses mRNA with 19nt window)
python scripts/main.py --infer 1 -i1 data/example.fa

# Inference Mode 2: Input mRNA and specific siRNA FASTA files
python scripts/main.py --infer 1 -i1 data/example.fa -i2 data/example_siRNA.fa

# Inference Mode 3: Manual mRNA sequence input
python scripts/main.py --infer 2

# With off-target prediction
python scripts/main.py --infer 1 -i1 ./data/example.fa -off -tox -a

# With off-target prediction, return top 100 siRNAs
python scripts/main.py --infer 1 -i1 ./data/example.fa -off -tox -a -top 100
```

**Key Arguments:**

| Argument | Description | Default |
|---------|-------------|---------|
| `--infer` | Inference mode (0=train, 1=infer from file, 2=manual input) | 0 |
| `-i1` | Input mRNA FASTA file | ./data/example.fa |
| `-i2` | Input siRNA FASTA file (optional) | - |
| `-off` | Enable off-target prediction | False |
| `-tox` | Enable toxicity prediction | False |
| `-top` | Top N siRNAs to analyze | -1 (all) |
| `-a` | Analyze all predicted siRNAs | False |

#### Result Post-processing

- **Output files**: Predictions saved to output directory
  - siRNA efficacy scores (0-1 scale, higher = more effective)
  - Off-target scores (if `-off` enabled)
  - Toxicity scores (if `-tox` enabled)
  - Ranked list of siRNA candidates with scores
- **Top siRNA selection**: Use `-top N` to get the top N most effective siRNAs
- **Sequence extraction**: Model outputs 19nt siRNA sequences with position information on mRNA

---

## Reference Resources

- **GitCode (primary)**: https://gitcode.com/AI4Science/OligoFormer
- **Official README**: https://gitcode.com/AI4Science/OligoFormer/blob/main/README.md
- **Additional reference**: https://github.com/lulab/OligoFormer
- **RNA-FM Repository**: `https://github.com/ml4bio/RNA-FM` - RNA feature extraction model
- **Paper**: OligoFormer: Transformer-based siRNA efficacy prediction (Tsinghua University)
- **Docker Image**: `yilanbai/oligoformer:v1.0` - Pre-built containerized version
- **Model Weights**: Available from GitHub releases and Google Drive