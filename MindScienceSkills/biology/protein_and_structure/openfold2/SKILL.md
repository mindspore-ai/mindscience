---
name: openfold2
description: openfold2 is a PyTorch reimplementation of AlphaFold 2 for protein structure prediction. Use this model when you need to train or deploy openfold2 on Ascend, or predict protein 3D structures from sequence.
license: Apache License 2.0
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend
---

# OpenFold

## Overview

OpenFold is a PyTorch reimplementation and extension of AlphaFold 2. It predicts 3D protein structures from amino acid sequences using attention-based models and supports training pipelines used in the Ascend ModelZoo packaging.

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

- **Scenario 1**: Train OpenFold on Ascend NPUs using the ModelZoo scripts and PDB alignment data from the public OpenFold S3 layout.
- **Scenario 2**: Validate trained checkpoints with the bundled 8-device validation script.
- **Scenario 3**: Protein structure prediction (inference) using the upstream OpenFold workflow and pretrained weights (see the reference repository).

---

## How It Works

### 1. Training environment

#### Supported versions

**Table 1 — Framework and Python**

| Torch_Version | Python |
|:-------------:|:------:|
| PyTorch 2.1   | 3.9    |

#### Host preparation

Follow Huawei’s guide for PyTorch training on Ascend: [PyTorch framework training environment preparation](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes).

#### Install Python dependencies

At the **root of the model source package**, install the requirements file that matches your PyTorch version (only one of these is needed):

```bash
pip install -r 2.1_requirements.txt   # PyTorch 2.1
```

Additional dependencies:

```bash
pip install git+https://github.com/NVIDIA/dllogger.git
pip install torch==2.1.0

wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-AVX2-Linux.tar.gz
tar xvfz hhsuite-3.3.0-AVX2-Linux.tar.gz
export PATH="$(pwd)/bin:$(pwd)/scripts:$PATH"

pip install git+https://github.com/TimoLassmann/kalign.git

wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
tar xvfz mmseqs-linux-avx2.tar.gz
export PATH=$(pwd)/mmseqs/bin/:$PATH
```

#### Build and install OpenFold

```bash
bash scripts/install_third_party_dependencies.sh
python setup.py install
```

---

### 2. Dataset preparation (training)

Assume the OpenFold repository root is `$OF_DIR` (set this to your actual clone path).

#### Install AWS CLI

Example for Linux aarch64 (use the [AWS CLI install page](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) for other platforms):

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

#### Download alignments and mmCIF archives

```bash
mkdir -p alignment_data/alignment_dir_roda
aws s3 cp s3://openfold/pdb/ alignment_data/alignment_dir_roda/ --recursive --no-sign-request

mkdir pdb_data
aws s3 cp s3://openfold/pdb_mmcif.zip pdb_data/ --no-sign-request
aws s3 cp s3://openfold/duplicate_pdb_chains.txt . --no-sign-request
unzip pdb_mmcif.zip -d pdb_data
```

Flatten the alignment directory layout to match OpenFold expectations:

```bash
bash $OF_DIR/scripts/flatten_roda.sh alignment_data/alignment_dir_roda alignment_data/
rm -r alignment_data/alignment_dir_roda
```

#### Create sharded alignment database

```bash
python $OF_DIR/scripts/alignment_db_scripts/create_alignment_db_sharded.py \
    alignment_data/alignments \
    alignment_data/alignment_dbs \
    alignment_db \
    --n_shards 10 \
    --duplicate_chains_file pdb_data/duplicate_pdb_chains.txt
```

Optional sanity check (expected line count **634434**):

```bash
grep "files" alignment_data/alignment_dbs/alignment_db.index | wc -l
```

#### Download PDB caches

```bash
aws s3 cp s3://openfold/data_caches/ pdb_data/ --recursive --no-sign-request
```

---

### 3. Training

#### Enter the source tree

```bash
cd /path/to/<model_folder_name>
```

#### 8-device training

```bash
bash test/train_openfold_8p.sh --data_path=<training_data_path>
```

#### 8-device validation (after training)

```bash
bash test/val_openfold_8p.sh \
  --data_path=<training_data_path> \
  --val_alignment_dir=<val_alignments_dir> \
  --val_data_dir=<val_mmcif_dir>
```

#### Common script arguments

| Argument | Meaning |
| -------- | ------- |
| `--data_path` | Training dataset path (required for training) |
| `--val_data_dir` | Directory with validation mmCIF files (required for validation) |
| `--val_alignment_dir` | Directory with validation alignments (required for validation) |
| `--max_epochs` | Number of training epochs (default: 1) |

Checkpoints are written under `output/checkpoints/`. Run the validation script after training to obtain validation metrics.

---

### 4. Reported training and validation results

**Table 2 — Example 8-device runs (bf16)**

| NAME | MODE | training_time | val/loss | Torch_Version |
|:----:|:----:|:-------------:|:--------:|:-------------:|
| 8p-vendor-A | bf16 | 1:38:03 | 78.30 | 2.1 |
| 8p-Atlas 900 A2 PODc | bf16 | 2:37:31 | 79.32 | 2.1 |

---

### 5. Inference (upstream reference)

For pretrained inference workflows (FASTA input, public weights, `run_pretrained_openfold.py`, etc.), use the [reference OpenFold repository](https://github.com/aqlaboratory/openfold) and its documentation. The Ascend ModelZoo package above is oriented toward **training** on Ascend with the listed scripts.

---

## Public network endpoints

Third-party URLs used by scripts and downloads are summarized in **`public_address_statement.md`** inside the model package. Review it for compliance with your network policy.

---

## Hardware and disk (training-oriented)

| Resource | Notes |
|----------|--------|
| NPU | Huawei Ascend (8-device scripts target 8 NPUs) |
| RAM | Large host RAM recommended for OpenFold-scale training |
| Storage | Substantial space for PDB mmCIF, alignments, caches, and checkpoints (plan for hundreds of GB) |

---

## End-to-end checklist (Ascend training)

- [ ] Obtain the Ascend ModelZoo OpenFold sources under `built-in/PyTorch/built-in/others` (or align your tree with the reference commit at the end of **Reference resources**).
- [ ] Prepare the Ascend PyTorch runtime per the HiAscend ModelZoo guide.
- [ ] Python 3.9; install `2.1_requirements.txt`, dllogger, `torch==2.1.0`, HH-suite, kalign, mmseqs, then `install_third_party_dependencies.sh` and `python setup.py install`.
- [ ] Install AWS CLI; download OpenFold S3 data; run `flatten_roda.sh`; build sharded alignment DB; download `data_caches` into `pdb_data/`.
- [ ] Run `test/train_openfold_8p.sh --data_path=...`.
- [ ] Run `test/val_openfold_8p.sh` with validation directories.
- [ ] Read `public_address_statement.md` for outbound URL policy.

---

## Model limitations

- Training is resource-heavy and assumes the full PDB alignment and cache layout described above.
- Sequence length and batch settings follow upstream OpenFold constraints.
- Inference accuracy depends on MSA and template databases when using full pipelines.

---

## Reference resources

- **Ascend ModelZoo (PyTorch)**: https://gitee.com/ascend/ModelZoo-PyTorch — path `built-in/PyTorch/built-in/others`
- **Reference OpenFold**: https://github.com/aqlaboratory/openfold (commit `e8d355874c3cc767e56af983d4e9a5190918eb6c`)
- **HiAscend PyTorch environment**: https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes
- **OpenFold S3 bucket**: `s3://openfold/` (public, no-sign-request)
- **AlphaFold 2**: https://www.nature.com/articles/s41586-021-03819-2
- **OpenFold preprint**: https://www.biorxiv.org/content/10.1101/2021.11.08.467743

### Reference implementation

```
url=https://github.com/aqlaboratory/openfold.git
commit_id=e8d355874c3cc767e56af983d4e9a5190918eb6c
```

### Ascend-adapted package

```
url=https://gitee.com/ascend/ModelZoo-PyTorch
code_path=built-in/PyTorch/built-in/others
```
