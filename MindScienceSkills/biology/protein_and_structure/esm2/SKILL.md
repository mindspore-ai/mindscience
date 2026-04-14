---
name: esm2
description: esm2 (Evolutionary Scale Modeling 2) is a large-scale protein language model developed by Meta AI. It uses transformer-based architecture with attention mechanisms to learn interactions between amino acid pairs in input sequences. Use this model when you need to extract protein sequence embeddings, predict contact maps, or perform protein structure-related inference tasks.
license: MIT
metadata:
    skill-author: MindSpore Science Team

---

# ESM2

## Overview

ESM2 is a transformer-based protein language model trained on protein sequence data at the scale of evolution. It represents the largest protein language model trained to date, with up to 15B parameters. ESM2 significantly outperforms previous models like ESM-1b—even the 150M parameter ESM2 model generates more accurate structural information than the 650M parameter ESM-1b model.

The model uses attention mechanisms to learn pairwise interactions between amino acids, enabling it to capture evolutionary relationships and predict protein structure features. ESM2 introduces relative position embeddings that can generalize to arbitrary sequence lengths.

This skill provides inference capabilities for ESM2 on Ascend NPU using MindSpore framework.

---

## When to Use

- **Protein structure prediction**: Extract contact maps and structural features from protein sequences
- **Protein embedding extraction**: Generate dense vector representations of protein sequences for downstream tasks
- **Protein family analysis**: Analyze evolutionary relationships across protein families
- **Masked sequence modeling**: Predict masked amino acids in protein sequences (e.g., "KA<mask>ISQ")
- **Feature extraction for downstream ML**: Use ESM2 embeddings as features for classification or regression tasks

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | Protein sequences in FASTA format or raw sequence strings   |
| Data Size   | Single sequences or multiple sequences (batch processing supported) |
| Data Source | UniProtKB, Pfam, or custom protein sequence databases       |

#### Data Acquisition Methods

1. **UniProt Download**: Download protein sequences from https://www.uniprot.org/
2. **Pfam Database**: Use protein family sequences from https://pfam.xfam.org/
3. **Custom Sequences**: Provide your own protein sequences in FASTA format

#### Data Preprocessing

- Ensure sequences use standard 20 amino acid letters (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
- Handle unknown or non-standard amino acids (X, B, Z) appropriately
- Mask specific positions using `<mask>` token for masked prediction tasks
- Sequence length should be within model's supported context window

---

### 2. Environment Configuration and Dependencies

#### Verified Ascend stack (reference)

| Component | Version     |
| --------- | ----------- |
| HDK       | -           |
| CANN      | 8.0.RC3     |
| Python    | -           |
| MindSpore | 2.4.0       |

Install **CANN** on the host per Huawei documentation before the Python steps.

#### Clone MindSpore Science repository and prepare environment

From a working directory of your choice:

```bash
git clone -b r0.7 https://gitcode.com/mindspore-lab/mindscience.git
cd mindscience
```

#### Conda environment and Python dependencies

```bash
conda create --name esm2 python=3.10
conda activate esm2
pip install mindspore=2.4.0
```

#### Environment Requirements (hardware / disk)

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Hardware    | Ascend NPU (910 series recommended)            |
| Memory      | At least 16GB RAM recommended                  |
| Disk Space  | At least 10GB for model checkpoints and outputs |

#### End-to-end checklist

| Step | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 1    | Clone **mindscience** repository with `git clone -b r0.7 https://gitcode.com/mindspore-lab/mindscience.git`. |
| 2    | Install CANN 8.0.RC3; create conda env `esm2` (Python 3.10); `pip install mindspore=2.4.0`. |
| 3    | **Inference**: Use the provided example code to run ESM2 inference. |

Use Linux or WSL (or Git Bash on Windows) so `bash` is available.

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type         | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| Functional Limitations  | Only inference is supported; training is not available in current pipeline |
| Performance Limitations | Inference time scales with sequence length and model size |
| Scale Limitations       | Different model sizes available: 8M, 35M, 150M, 3B, 15B parameters |
| Input Format            | Standard 20 amino acid letters; use `<mask>` for masked prediction |

#### Notes

- **Note 1**: ESM2 in MindSpore pipeline currently supports inference only. Training functionality is not available.
- **Note 2**: The model uses relative position embeddings that can generalize to arbitrary sequence lengths.
- **Note 3**: ESM2 with 150M parameters outperforms ESM-1b with 650M parameters on structure prediction benchmarks.
- **Note 4**: Pre-training data is UR50/D 2021_04 dataset.

---

### 4. Model Invocation Guide

#### Model Initialization

The ESM2 model is accessed through the MindSponge pipeline:

```python
from mindsponge.pipeline import PipeLine

pipeline = PipeLine('ESM2')
pipeline.initialize('config')
pipeline.model.from_pretrained()
```

#### Running Inference

**Python:**

```python
import numpy as np
from mindsponge.pipeline import PipeLine

# Initialize pipeline
pipeline = PipeLine('ESM2')
pipeline.initialize('config')
pipeline.model.from_pretrained()

# Prepare input data (sequence with mask)
data = [("protein3", "KA<mask>ISQ")]

# Run prediction
kwargs = {"return_contacts": True}
_, _, _, contacts = pipeline.predict(data, **kwargs)

# Process results
contacts = contacts.asnumpy()
tokens_len = pipeline.dataset.batch_lens[0]
attention_contacts = contacts[0]
matrix = attention_contacts[:tokens_len, :tokens_len]
print("contact map", matrix)
```

#### Result Post-processing

- **Contact maps**: Returned as numpy arrays, can be used for protein structure prediction
- **Embeddings**: Can be extracted from intermediate layers for downstream tasks
- **Masked predictions**: Use `<mask>` token in input sequence to predict masked positions

---

## Reference Resources

- **MindSpore Science (ESM2)**: https://gitcode.com/mindspore-lab/mindscience/tree/r0.7
- **ESM2 Paper**: https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf
- **UniProt**: https://www.uniprot.org/
- **Pfam Database**: https://pfam.xfam.org/

## Citation

```bash
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```