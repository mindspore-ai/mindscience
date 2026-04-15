---
name: chemprop
description: chemprop is a message passing neural network (MPNN) framework for molecular property prediction from SMILES. This skill documents Ascend 910B deployment via CANN Docker, torch/torch_npu wheels, and training with train.py on the AtomGit fork.
license: MIT
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend
---

# ChemProp

## Overview

ChemProp is a deep learning framework for molecular property prediction using Message Passing Neural Networks (MPNNs). It predicts chemical properties (e.g., solubility, toxicity, lipophilicity, ADMET metrics) directly from molecular structures represented as SMILES strings. The model employs a graph neural network architecture that treats molecules as graphs with atoms as nodes and bonds as edges, enabling learned representations that capture molecular structure-property relationships.

ChemProp is widely used in drug discovery, materials science, and computational chemistry for high-throughput screening of molecular candidates.

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

- **Scenario 1**: Molecular property prediction - Suitable for predicting continuous values (regression) such as logP, solubility, melting point
- **Scenario 2**: Binary classification - Suitable for predicting binary outcomes such as toxicity, activity, hERG blockade
- **Scenario 3**: Multi-class classification - Suitable for predicting categorical outcomes such as compound type or reaction class
- **Scenario 4**: Uncertainty quantification - Suitable for obtaining prediction confidence using ensemble, dropout, or evidential methods

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | CSV files with SMILES strings |
| Data Size | Recommended 100+ molecules for training; can work with smaller datasets |
| Data Source | Public datasets (e.g., QM9, ESOL, Lipophilicity), in-house data, or literature |

#### Data Acquisition Methods

1. **Public Molecular Databases** - Download from sources like PubChem, ChEMBL, or ZINC for specific property datasets
2. **Benchmark Datasets** - Use established benchmarks: ESOL (solubility), Lipophilicity, Tox21 (toxicity), QM9 (quantum mechanical properties)
3. **Custom CSV** - Create your own dataset with SMILES and target properties

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Prepare a CSV file with SMILES in the first column (or specify column name with `-s`)
- **Step 2**: For training, include target values in additional columns
- **Step 3**: Ensure SMILES are valid (can use RDKit to validate)
- **Step 4**: For regression tasks, consider whether targets need normalization

**Example input CSV format:**
```csv
smiles,logSolubility
OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O,-0.77
Cc1occc1C(=O)Nc2ccccc2,-3.3
CCO,0.65
```

---

### 2. Environment Configuration and Dependencies

#### Component versions

| Component | Version |
| --------- | ------------------------------ |
| HDK       | 25.2.0 |
| CANN      | 8.3.rc1 |
| Python    | 3.11.13 |
| torch     | 2.1.0 |
| torch-npu | 2.1.0.post17 |

```shell
docker run -it -u root \
  --net=host --shm-size=5g \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
  -v /home:/home/ \
  --name chemprop_test \
  --entrypoint=/bin/bash \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-910b-ubuntu22.04-py3.11
```

Image catalog: [Ascend Hub (Huawei)](https://www.hiascend.com/developer/ascendhub/). Adjust `--device` entries (`davinci*`) to match the number of NPUs on your machine.

#### Install torch and torch_npu

**Download wheels (aarch64, cp311):**

```shell
wget https://download.pytorch.org/whl/cpu/torch-2.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
wget https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.1.0/torch_npu-2.1.0.post17-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

**Install:**

```shell
pip3 install torch-2.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch_npu-2.1.0.post17-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

#### System packages

```shell
apt update && apt install -y libsm6 libxext6
```

#### Other Python dependencies

```shell
pip3 install astartes aimsim configargparse "lightning>=2.0" scikit-learn==1.1.3 descriptastorus rich cloudpickle ml-dtypes tornado numpy==1.26.4
```

Pin **`cloudpickle==1.6.1`** if you need an exact match to the component list above.

#### Clone and install ChemProp

```shell
git clone https://atomgit.com/AI4Science/chemprop.git
cd chemprop && pip install -e .
```

#### Environment requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend 910B (example image above) |
| Memory | 8GB+ host RAM typical; increase `--shm-size` if DataLoader workers complain |
| Disk | Wheels, repo, and checkpoint directory |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | Start CANN container with host driver/NPU devices mounted |
| 2 | Install **torch** and **torch_npu** wheels; `apt` libs; `pip3` dependency line |
| 3 | Clone **atomgit** `chemprop`, `pip install -e .` |
| 4 | Prepare CSV with SMILES (and targets) |
| 5 | `source` Ascend toolkit env; run **`train.py`** (below) |
| 6 | Optional: use upstream **chemprop** CLI for predict/train if your tree matches that API |

#### Training

Load the toolkit environment (path may be `ascend-toolkit` or `ascend_toolkit` depending on install):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# alternate: source /usr/local/Ascend/ascend_toolkit/set_env.sh
```

Example (built-in test CSV):

```bash
python3 train.py --data_path tests/data/classification.csv --dataset_type classification --save_dir classification_checkpoints
```

General form:

```bash
python3 train.py --data_path <path> --dataset_type <type> --save_dir <dir>
```

| Argument | Description |
| -------- | ----------- |
| `<path>` | Path to your CSV dataset |
| `<type>` | One of **`classification`**, **`regression`**, **`multiclass`**, **`spectra`** |
| `<dir>` | Directory where checkpoints / training outputs are saved |

Run `train.py` from the **chemprop** repository root after `pip install -e .`.

---

### 3. Usage limitations and notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Functional Limitations | Cannot predict 3D molecular conformations; only works with 2D molecular graphs |
| Performance Limitations | Performance depends on training data quality and quantity; may underperform on out-of-distribution molecules |
| Scale Limitations | Very large molecules (>500 atoms) may cause memory issues; batch size may need reduction |
| Input Format | Requires valid SMILES strings; invalid or unknown molecules will cause errors |

#### Notes

- **Note 1**: ChemProp requires valid SMILES. Use RDKit to validate SMILES before prediction.
- **Note 2**: For best results, use a training dataset similar to your target molecules in chemical space.
- **Note 3**: Pre-trained models can be downloaded from Zenodo (e.g., the Halicin discovery models: https://doi.org/10.5281/zenodo.6527882)
- **Note 4**: The model supports uncertainty quantification via multiple methods (ensemble, dropout, evidential, MVE). Use `--uncertainty-method` to enable.

---

### 4. Model invocation guide

#### Model Initialization

ChemProp does not require explicit model initialization for inference — the model checkpoint is loaded via CLI arguments.

| Item | Example / value |
| ---- | ---------------- |
| Model checkpoint | `.ckpt` or `.pt` file from training |
| Input format | CSV with SMILES column |

#### Running examples (recommended path)

**CLI Prediction (using a trained model):**

```bash
# Basic prediction
chemprop predict \
    --test-path your_molecules.csv \
    --model-paths trained_model.ckpt \
    --preds-path predictions.csv
```

**CLI Training:**

```bash
chemprop train \
    --data-path train.csv \
    --task-type regression \
    --output-dir output_folder
```

**Python API (optional):**

```python
from chemprop import predict
from chemprop.ensemble import EnsemblePredict

# Single model prediction
predictions = predict.predict(
    test_path="molecules.csv",
    model_paths=["model.ckpt"]
)

# Ensemble prediction
ensemble = EnsemblePredict(model_paths=["model1.ckpt", "model2.ckpt"])
predictions = ensemble.predict(test_path="molecules.csv")
```

#### Result Post-processing

- Predictions are saved to the specified output path (CSV format by default)
- For multi-task models, each target gets its own column
- With ensemble models, predictions are averaged; individual model predictions saved to `_individual` file
- Uncertainty estimates (if enabled) are saved as additional columns

---

## Reference resources

- **AtomGit (clone URL)**: https://atomgit.com/AI4Science/chemprop
- **GitCode mirror**: https://gitcode.com/AI4Science/ChemProp
- **Upstream ChemProp**: https://github.com/chemprop/chemprop
- **Ascend Hub**: https://www.hiascend.com/developer/ascendhub/
- **Documentation**: `https://chemprop.readthedocs.io/`
- **Pre-trained Models**: https://doi.org/10.5281/zenodo.6527882