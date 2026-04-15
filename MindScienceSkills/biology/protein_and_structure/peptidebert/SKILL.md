---
name: peptidebert
description: peptidebert is a transformer-based deep learning model for peptide property prediction. It leverages the pre-trained ProtBERT-BFD model as a backbone with a custom classification head for binary prediction tasks. Use this model when you need to predict peptide properties such as hemolysis, solubility, or non-fouling behavior.
license: MIT
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend
---

# PeptideBERT

## Overview

PeptideBERT is a transformer-based language model designed for peptide property prediction. It leverages the pre-trained **ProtBERT-BFD** model (`Rostlab/prot_bert_bfd`) as a backbone and adds a custom classification head for binary prediction tasks. The model can predict whether peptides exhibit specific properties such as causing hemolysis (red blood cell lysis), solubility, or non-fouling behavior.

The model uses a message passing architecture combined with attention mechanisms to capture both sequence and structural features of peptides, enabling high-accuracy property predictions for various biotechnology and pharmaceutical applications.

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

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits your task requirements.

- **Scenario 1**: Hemolysis prediction - Suitable for predicting whether peptide sequences cause red blood cell lysis, important for therapeutic peptide safety assessment
- **Scenario 2**: Solubility prediction - Suitable for predicting peptide solubility, critical for drug formulation and delivery
- **Scenario 3**: Non-fouling property prediction - Suitable for predicting non-fouling behavior, useful for biomaterial surface design

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | FASTA format (one sequence per line), NPZ files for training |
| Data Size | Training: ~2000-5000 peptides per class; Inference: any number of sequences |
| Data Source | Peptide property datasets from https://github.com/ur-whitelab/peptide-dashboard |

#### Data Acquisition Methods

1. **Automatic Download** - Run `python data/download_data.py` to download datasets from the peptide-dashboard repository
2. **Manual Download** - Download NPZ files from the source repository and place in appropriate directories

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Ensure peptide sequences use standard amino acid single-letter codes (L, A, G, V, E, S, I, K, R, D, T, P, N, Q, F, Y, M, H, C, W)
- **Step 2**: For inference, create input file with one peptide sequence per line (FASTA-like format)
- **Step 3**: Sequences will be tokenized using the model's vocabulary mapping
- **Step 4**: Pad sequences to maximum length in the batch for batch processing

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Verified Ascend stack (reference)

| Component | Version |
| --------- | ------------------------------- |
| HDK       | 24.1.RC3 |
| CANN      | 8.0.RC3 |
| Python    | 3.9 |
| torch     | 2.1.0 |
| torch-npu | 2.1.0.post8 |

#### Clone repository and prepare code

```bash
git clone https://ai.gitcode.com/AI4Science/PeptideBERT.git
# Or: https://gitcode.com/AI4Science/PeptideBERT.git
cd PeptideBERT
```

#### Conda environment and Python dependencies

```bash
# Create conda environment
conda create --name peptidebert python=3.9
conda activate peptidebert

# Navigate to project directory
cd /path/to/PeptideBERT

# Install dependencies
pip install -r requirements.txt
```

**For Ascend NPU — additional torch-npu installation:**

After installing requirements.txt, install the NPU-specific torch build:

```bash
# Install torch 2.1.0
pip install torch==2.1.0

# Download and install torch-npu wheel (matching CANN 8.0.RC3)
wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.1.0/torch_npu-2.1.0.post8-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install torch_npu-2.1.0.post8-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

**Set up CANN environment (if using Ascend NPU):**
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

---

### 3. Running Inference

This module explains how to run inference with trained PeptideBERT models.

#### Prepare Input Data

Create a file `data/input.txt` with one peptide sequence per line:

```
LLKKLLKKLLKK
AAAGGGVVV
RKLLK
```

#### Run Inference

```python
import torch
import yaml
from model.network import create_model

# Configuration
run_name = 'YOUR_CHECKPOINT_FOLDER'  # e.g., 'sol-1230_1430'
device = torch.device(
    'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'
)

config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device

# Create and load model
model = create_model(config)
model.load_state_dict(torch.load(f'./checkpoints/{run_name}/model.pt')['model_state_dict'], strict=False)
model.to(device)
model.eval()

# Read input sequences
seqs = []
with open('./data/input.txt', 'r') as f:
    for line in f.readlines():
        seq = line.strip()
        if seq:
            seqs.append(seq)

# Tokenize sequences
MAX_LEN = max(map(len, seqs))
mapping = dict(zip(
    ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','L',
    'A','G','V','E','S','I','K','R','D','T','P','N',
    'Q','F','Y','M','H','C','W'],
    range(30)
))

tokenized_seqs = []
for seq in seqs:
    tokenized = [mapping.get(c, mapping['[UNK]']) for c in seq]
    tokenized.extend([0] * (MAX_LEN - len(tokenized)))
    tokenized_seqs.append(tokenized)

# Run inference
preds = []
with torch.inference_mode():
    for i in range(len(tokenized_seqs)):
        input_ids = torch.tensor([tokenized_seqs[i]]).to(device)
        attention_mask = (input_ids != 0).float()
        output = model(input_ids, attention_mask)
        prediction = int(output.item() > 0.5)
        preds.append(prediction)

# Save predictions
with open('./data/output.txt', 'w') as f:
    for pred in preds:
        f.write(str(pred) + '\n')

print(f"Predictions saved to data/output.txt")
```

#### Training (for reference)

To train a model, edit `config.yaml` to set the task (`hemo`, `sol`, or `nf`), then run:

```bash
# Download datasets
python data/download_data.py

# Split and prepare data
python data/split_augment.py

# Train model
python train.py
```

---

### 4. Model Checkpoints

Model checkpoints are saved in the `checkpoints/` directory. Training produces:
- `model.pt` - Model weights and state
- Training logs and metrics

Pre-trained checkpoints for specific tasks can be obtained by training the model or from the model authors.

---

## Reference Resources

| Resource | URL |
|----------|-----|
| GitCode (primary) | https://ai.gitcode.com/AI4Science/PeptideBERT |
| Official README | https://ai.gitcode.com/AI4Science/PeptideBERT/blob/main/README.md |
| Additional reference | https://github.com/ChakradharG/PeptideBERT |
| Pre-trained Backbone | Rostlab/prot_bert_bfd (HuggingFace) |
| Data Source | https://github.com/ur-whitelab/peptide-dashboard |

---

## End-to-end Checklist

- [ ] Clone repository (`git clone https://ai.gitcode.com/AI4Science/PeptideBERT.git`)
- [ ] Create conda environment with Python 3.9
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] For Ascend NPU: Install torch-npu wheel matching CANN version
- [ ] Prepare input sequences in `data/input.txt`
- [ ] Load trained checkpoint or train new model
- [ ] Run inference and obtain predictions
- [ ] Check output in `data/output.txt`

---

## Model Limitations

- **Sequence Length**: Maximum sequence length is determined by model configuration; very long peptides may need truncation
- **Vocabulary**: Only standard 20 amino acids are supported; non-standard residues may be treated as unknown
- **Task Specific**: Model must be trained or fine-tuned for each property prediction task (hemolysis, solubility, non-fouling)
- **Binary Classification**: Model outputs binary predictions; probability thresholds may need adjustment for specific applications