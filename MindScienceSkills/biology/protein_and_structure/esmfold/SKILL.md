---
name: esmfold
description: esmfold is a deep learning model for protein structure prediction developed by Meta AI (Facebook Research). It uses esm-2 protein language model as a backbone with a folding module to predict 3D protein structures directly from amino acid sequences. Use this model when you need to predict protein structures from sequences without requiring multiple sequence alignments.
license: MIT
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend
---

# ESMfold

## Overview

ESMfold (Evolutionary Scale Modeling fold) is an end-to-end protein structure prediction model developed by Meta AI (Facebook Research). It leverages the ESM-2 protein language model as a learned representation and combines it with a folding module to directly predict 3D protein structures from amino acid sequences.

The model represents a paradigm shift in structure prediction by using a single-sequence approach (no MSA required), making it significantly faster than template-based methods while achieving competitive accuracy on benchmark datasets.

**Available Models:**

| Model | Parameters | Description |
|-------|------------|-------------|
| esmfold_v0 | ~690M | Original model from the paper (Lin et al, 2022). Trained on PDB chains until 2020-05. |
| esmfold_v1 | ~690M + 3B ESM-2 | **Recommended version**. Uses 3B ESM-2 with 48 folding blocks. Higher accuracy and faster inference. |

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

- **Scenario 1**: Fast protein structure prediction - Suitable for predicting 3D protein structures from amino acid sequences without MSA generation
- **Scenario 2**: High-throughput structure prediction - Suitable for screening large numbers of protein sequences
- **Scenario 3**: Confidence scoring - Suitable for obtaining per-residue pLDDT confidence scores and predicted TM-scores
- **Scenario 4**: Multimer prediction - Suitable for predicting structures of protein complexes (chains separated by ":")

---

## How It Works

### 1. Dataset acquisition and processing

| Requirement | Description |
| ----------- | ----------- |
| Format | FASTA or plain one-letter amino acid sequences |
| Length | Roughly up to ~2000 residues; longer inputs may need chunking or may OOM |
| Charset | Standard letters `ACDEFGHIKLMNPQRSTVWY`; nonstandard symbols may hurt quality |

- Multimer: `chain1:chain2` in one sequence string.
- Avoid unknowns where possible (`X`, etc.).

---

### 2. Environment configuration and dependencies

#### Component versions

```shell
hdk: 24.1.0.3
cann: 8.0.RC3
python: 3.9.2
torch: 2.1.0
torch_npu: 2.1.0.post8
```

#### Install fair-esm / ESMFold entry points

```bash
pip install fair-esm
pip install "fair-esm[esmfold]"
```

#### Install PyTorch and torch_npu (Ascend)

```bash
pip install torch==2.1.0
wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.1.0/torch_npu-2.1.0.post8-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

If TLS certificate verification fails, append `--no-check-certificate` to `wget`, then:

```bash
pip3 install torch_npu-2.1.0.post8-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

Use a `torch_npu` wheel that matches your **Python ABI** (cp39) and **CANN** release if you deviate from the table above.

#### Install OpenFold (NPU / vendor build)

Follow the instructions in the Ascend-oriented OpenFold guide:

https://modelers.cn/models/Ascend-AI4S/OpenFold1.0.0/blob/main/README.md

Complete that build before relying on folding with the patched runtime below.

#### Example FASTA (`test.fasta`)

Create a minimal FASTA file:

```bash
touch test.fasta
# edit test.fasta to contain:
```

```fasta
>test
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
```

#### Download checkpoints

```bash
wget https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt
wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt
```

Move the files into the Torch hub checkpoint directory, for example:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
mv esmfold_3B_v1.pt esm2_t36_3B_UR50D.pt esm2_t36_3B_UR50D-contact-regression.pt ~/.cache/torch/hub/checkpoints/
```

(Documentation sometimes uses `/root/.cache/torch/hub/checkpoints/` when running as root; same layout under the active user’s home.)

#### Enable torch_npu in the ESMFold CLI script

Edit `esm/scripts/fold.py` inside your **installed** `fair-esm` package (path depends on prefix, e.g. `.../site-packages/esm/scripts/fold.py`). Immediately after the existing `torch` import, add:

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

(Example path from deployment guides: `/usr/local/python3.9.2/lib/python3.9/site-packages/esm/scripts/fold.py` — adjust to your Python location.)

#### Environment requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend NPU |
| Memory | Large device memory recommended for 3B ESMFold; ample host RAM |
| Disk | Several GB for listed checkpoints and OpenFold build artifacts |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | Match **Component versions**; install **fair-esm** / **fair-esm[esmfold]** |
| 2 | Install **torch==2.1.0** and **torch_npu** wheel (Gitee link above or equivalent) |
| 3 | Build/install **OpenFold** per the modelers.cn README |
| 4 | Place checkpoints under `~/.cache/torch/hub/checkpoints/` |
| 5 | Patch **`esm/scripts/fold.py`** with `torch_npu` imports |
| 6 | `source` Ascend toolkit env; run **`esm-fold`** (below) |

**Optional NPU check:**

```bash
python -c "import torch; import torch_npu; print(torch.__version__); print(torch_npu.npu.is_available())"
```

---

### 3. Inference (Ascend)

Source the toolkit environment (path may differ by install):

```bash
source /usr/local/Ascend/ascend_toolkit/set_env.sh
esm-fold -i test.fasta -o ./
```

First runs may be slow while libraries initialize.

**CLI reference (same tool):**

```bash
esm-fold -i input.fasta -o output_directory/
esm-fold -i proteins.fasta -o pdb_output/ \
    --num-recycles 4 \
    --max-tokens-per-batch 1024 \
    --chunk-size 128
```

| Argument | Description | Default |
| -------- | ----------- | ------- |
| `-i`, `--fasta` | Input FASTA | required |
| `-o`, `--pdb` | Output directory for PDBs | required |
| `--num-recycles` | Recycling iterations | 4 |
| `--max-tokens-per-batch` | Token cap per forward | 1024 |
| `--chunk-size` | Axial-attention chunk size | auto |

---

### 4. Python API (optional)

```python
import torch
import esm

model = esm.pretrained.esmfold_v1()
model = model.eval()
if hasattr(torch, "npu") and torch.npu.is_available():
    model = model.to("npu")

model.set_chunk_size(128)
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

with torch.no_grad():
    output = model.infer_pdb(sequence)

with open("result.pdb", "w") as f:
    f.write(output)
```

For tensors and metadata (`plddt`, `ptm`, etc.), use `model.infer(sequence, num_recycles=4)` and inspect the returned dict.

---

### 5. Usage limitations and notes

| Topic | Notes |
| ----- | ----- |
| MSA-free | Weaker than MSA-heavy methods on some families |
| Memory | Lower `chunk_size` (e.g. 64) if you hit OOM |
| Length | Very long chains may need chunking or batching changes |
| Accuracy | Benchmarks vs AlphaFold2-class models vary by target class |

pLDDT is high when confident (e.g. above 90 is very strong); pTM-style scores above ~0.7 suggest reasonable global fold.

---

## Reference resources

- **OpenFold (Ascend guide)**: https://modelers.cn/models/Ascend-AI4S/OpenFold1.0.0/blob/main/README.md
- **fair-esm / ESMFold (upstream)**: https://github.com/facebookresearch/esm
- **GitCode mirror (if used)**: https://ai.gitcode.com/AI4Science/ESMfold
- **Weights**: `https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt`, `esm2_t36_3B_UR50D.pt`, regression checkpoint as above
- **Paper**: Lin et al., *Nature Biotechnology* (2022) — learned folding modules for structure prediction
