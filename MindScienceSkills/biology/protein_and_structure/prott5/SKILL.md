---
name: prott5
description: prott5 is a transformer-based protein language model from the ProtTrans project. It processes protein sequences using a T5 (Text-to-Text Transfer Transformer) architecture to generate embeddings and perform sequence generation tasks. Use this model when you need to extract protein sequence embeddings or generate protein sequences from embeddings on MindSpore.
license: Apache 2.0
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend

---

# ProtT5

## Overview

ProtT5 is a state-of-the-art protein language model developed by the ProtTrans project. It uses a T5 (Text-to-Text Transfer Transformer) architecture to process protein sequences, enabling tasks such as protein embedding extraction and sequence generation. The model is pre-trained on large protein sequence databases and can be fine-tuned for various downstream tasks including protein property prediction and sequence design.

This skill provides inference capabilities for ProtT5 on MindSpore, enabling users to generate protein sequence embeddings and perform sequence generation tasks.

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

- **Protein embedding extraction**: Generate dense vector representations of protein sequences for downstream machine learning tasks
- **Protein property prediction**: Use embeddings as features for classification or regression tasks (requires fine-tuning)
- **Sequence generation**: Generate protein sequences from embeddings or continue partial sequences
- **Protein language modeling**: Understand protein sequence patterns and evolutionary relationships

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | Protein sequences in FASTA format or space-separated amino acid sequences |
| Data Size   | Single sequences or batches of sequences                    |
| Data Source | UniProt, PDB, or custom protein sequence databases          |

#### Data Acquisition Methods

1. **UniProt Download**: Download protein sequences from https://www.uniprot.org/
2. **PDB Extraction**: Extract sequences from protein structure files (.pdb)
3. **Custom Sequences**: Provide space-separated amino acid sequences in Python lists

#### Data Preprocessing

- Input sequences should be space-separated (e.g., "A E T C Z A O")
- Unknown or non-standard amino acids are represented with special tokens (e.g., "Z", "O")
- Sequences can be of varying lengths; the model handles padding internally

---

### 2. Environment Configuration and Dependencies

#### Dependency Installation

```bash
# Clone the MindSpore Science repository
git clone -b legacy-master https://gitee.com/mindspore/mindscience.git
cd mindscience/MindSponge
pip install -r requirements.txt

# Set Python path
export PYTHONPATH=mindscience/MindSPONGE/src:${PYTHONPATH}
```

#### Environment Requirements

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Python Version | Python 3.8+                                  |
| Framework    | MindSpore >= 2.3.0, mindformers == 0.8.0    |
| Hardware     | Ascend NPU (910 series recommended) or CPU   |
| Memory       | At least 16GB RAM recommended                 |
| Disk Space   | At least 10GB for model checkpoints           |
| Other        | sentencepiece >= 0.2.0                        |

#### Model Checkpoint Setup

Download the model checkpoint:
```bash
wget https://download-mindspore.osinfra.cn/mindscience/mindsponge/ProtT5/checkpoint/prot_t5_xl.ckpt
```

Or convert from PyTorch weights:
```bash
# Download PyTorch weights from HuggingFace: https://huggingface.co/Rostlab/prot_t5_xl_uniref50
python scripts/convert_weight.py --layers 24 --torch_path pytorch_model.bin --mindspore_path ./mindspore_t5.ckpt
```

Required files structure:
```bash
└── prot_t5_xl_uniref50_ms
    ├── prot_t5_xl_uniref50.ckpt   # Weight file (converted from PyTorch)
    ├── prot_t5.yaml               # Network configuration file
    ├── special_tokens_map.json    # Tokenizer
    ├── spiece.model               # Tokenizer model
    └── tokenizer_config.json      # Tokenizer config
```

#### End-to-end checklist

| Step | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 1    | Clone MindSpore Science repository: `git clone -b legacy-master https://gitee.com/mindspore/mindscience.git` |
| 2    | Install dependencies: `pip install -r requirements.txt` in MindSponge directory |
| 3    | Set PYTHONPATH: `export PYTHONPATH=mindscience/MindSPONGE/src:${PYTHONPATH}` |
| 4    | Download or convert model checkpoint to MindSpore format   |
| 5    | Prepare tokenizer files (special_tokens_map.json, spiece.model, tokenizer_config.json) |
| 6    | Run inference using the provided Python code                |

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type         | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| Functional Limitations  | Requires MindSpore framework; limited to inference tasks   |
| Performance Limitations | Inference time scales with sequence length                 |
| Scale Limitations       | Maximum recommended sequence length ~2000 amino acids      |
| Input Format            | Must be valid amino acid sequences with standard or recognized tokens |

#### Notes

- **Note 1**: ProtT5 uses SentencePiece tokenization; ensure tokenizer files are properly configured
- **Note 2**: The model supports two inference modes: "generate" for sequence generation and "embedding" for extracting protein representations
- **Note 3**: Non-standard amino acids (Z, O, B) are treated as unknown tokens and replaced with "X"
- **Note 4**: For production deployment, ensure MindSpore >= 2.3.0 is installed for optimal performance

---

### 4. Model Invocation Guide

#### Model Initialization

```python
from mindsponge.common.config_load import load_config
from mindsponge import PipeLine
import mindspore as ms

# Configure pipeline
config_path = 'mindscience/MindSPONGE/applications/model_configs/ProtT5/t5_predict.yaml'
pipe = PipeLine(name="ProtT5")
pipe.set_device_id(0)
conf = load_config(config_path)
pipe.initialize(conf=conf)
```

#### Running Inference

**Sequence Generation (mask filling):**
```python
# Input: space-separated amino acid sequences
data = ["A E T C Z A O", "S K T Z P"]
res = pipe.predict(data, mode="generate")
print("Generated:", res)
# Output: ['A E T C X A X', 'S K T X P']
```

**Embedding Extraction:**
```python
# Extract protein sequence embeddings
data = ["A E T C Z A O", "S K T Z P"]
res = pipe.predict(data, mode="embedding")
print("Embedding:", res)
# Output: 3D numpy arrays with embedding vectors
```

#### Result Post-processing

- **Generate mode**: Returns sequences with unknown tokens replaced
- **Embedding mode**: Returns 3D arrays of shape (batch_size, sequence_length, embedding_dim)
- For downstream tasks, typically use mean pooling over the sequence dimension

---

## Reference Resources

- **ProtTrans GitHub**: https://github.com/agemagician/ProtTrans
- **MindSpore Checkpoint**: https://download-mindspore.osinfra.cn/mindscience/mindsponge/ProtT5/checkpoint/
- **HuggingFace Model**: https://huggingface.co/Rostlab/prot_t5_xl_uniref50
- **MindSpore Documentation**: https://www.mindspore.cn/
- **MindSponge Documentation**: https://gitee.com/mindspore/mindscience/tree/master/MindSponge