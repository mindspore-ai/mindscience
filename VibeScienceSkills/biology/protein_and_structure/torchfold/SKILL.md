---
name: torchfold
description: torchfold is a PyTorch reimplementation of AlphaFold 3 for protein structure prediction. Use this model when you need to predict the 3D structure of proteins and other molecular complexes (DNA, RNA, ligands) from sequence information.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# TorchFold

## Overview

TorchFold is a PyTorch reimplementation of AlphaFold 3, originally developed by Google DeepMind. It predicts the 3D structure of proteins and other molecular complexes including DNA, RNA, and ligands from sequence information. The model uses a diffusion-based approach to generate atomic-level structure predictions, similar to the original AlphaFold 3 architecture.

This implementation provides a pure PyTorch alternative to the JAX-based original, enabling easier integration with PyTorch-based workflows and potential hardware acceleration on various backends.

---

## When to Use

This module details the primary application scenarios and typical use cases of the model, helping users determine whether the model suits their task requirements.

- **Scenario 1**: Protein structure prediction - Suitable for predicting 3D structures of proteins from their amino acid sequences
- **Scenario 2**: Multi-chain complex prediction - Suitable for predicting structures of protein-protein, protein-DNA, protein-RNA, and protein-ligand complexes
- **Scenario 3**: Drug discovery support - Suitable for generating structural predictions of protein-ligand complexes to support virtual screening and drug design workflows

---

## How It Works

### 1. Dataset Acquisition and Processing

This module explains the data format requirements, acquisition methods, and preprocessing steps to ensure users can properly prepare input data.

#### Dataset Requirements

| Requirement | Description |
|-------------|--------------|
| Data Format | JSON files (AlphaFold3 dialect) containing sequence and structure definitions |
| Data Size | Single JSON files for individual predictions; supports batch processing |
| Data Source | User-provided JSON input following AlphaFold3 schema |

#### Data Acquisition Methods

1. **Manual JSON Creation** - Create input JSON files following the AlphaFold3 schema with protein sequences, chain IDs, and optional ligand/DNA/RNA definitions
2. **RCSB PDB Conversion** - Convert existing PDB files to AlphaFold3 JSON format using provided conversion tools

#### Data Preprocessing

Users need to preprocess data according to the following steps:

- **Step 1**: Prepare input JSON file with protein sequences in AlphaFold3 dialect format
- **Step 2**: Ensure sequences use standard amino acid one-letter codes
- **Step 3**: For complexes, specify chain IDs and molecule types (protein/RNA/DNA/ligand)
- **Step 4**: Place JSON file in an accessible directory path

**Example input JSON format:**
```json
{
  "name": "test",
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGHDHNMTYIQALRHFSTFANGLHLSKQPINNLAQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

---

### 2. Environment Configuration and Dependencies

This module describes the environment requirements, dependencies, and installation methods needed to run the model, helping users quickly set up the development environment.

#### Component versions

```shell
cann: 8.3.RC1.alpha001
python: 3.11
torch: 2.6.0
torch-npu: 2.6.0
```

#### Verified Ascend stack (reference)

**Note**: The official README documents **CANN**, **Python**, **torch**, and **torch_npu**. It does not state an **HDK** version explicitly; install Ascend driver and firmware compatible with **CANN 8.3.RC1.alpha001** per Huawei documentation.

| Component | Version |
| --------- | ----------------------------- |
| HDK       | must match CANN 8.3.RC1.alpha001 (see Huawei Ascend documentation) |
| CANN      | 8.3.RC1.alpha001 |
| Python    | 3.11 |
| torch     | 2.6.0 |
| torch-npu | 2.6.0 |

#### Clone repository

```bash
git clone https://gitcode.com/AI4Science/Mingchenchen.git
cd Mingchenchen
# Model code is under torchfold/.
```

#### Environment setup (full sequence)

**a. Create conda environment**

```shell
conda create --name torchfold python=3.11
conda activate torchfold
```

**b. Install HMMER**

```shell
conda install -c bioconda hmmer
```

**c. PyTorch / NPU and base Python packages**

```shell
pip install decorator attrs jinja2 psutil absl-py cloudpickle ml-dtypes scipy tornado pyyaml pybind11 loguru
pip install torch==2.6.0
pip install torch_npu==2.6.0
```

**d. Install mx_driving (DrivingSDK)**

After cloning, set `ENABLE_ONNX` to `False` in `CMakePresets.json`.

```shell
git clone https://gitee.com/ascend/DrivingSDK.git
cd DrivingSDK
pip install cmake   # requires >= 3.19.0
pip install -r requirements.txt
bash ci/build.sh --python=3.11
pip3 install dist/mx_driving-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl
cd ..
```

Replace `{commit_id}`, `{python_version}`, and `{arch}` with the wheel filename produced under `dist/`.

**e. Install triton-ascend**

Use **3.2.0.dev2025103116** or any release **after August 2025**.

- (1) Download the wheel from [triton-ascend (TestPyPI)](https://test.pypi.org/project/triton-ascend/#files) and install, for example:

```shell
pip install triton_ascend-3.2.0.dev2025103116-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
```

- (2) Or install from TestPyPI (if you hit SSLError, use Huawei gateway CA: append `--cert Huawei_Web_Secure_Internet_Gateway_CA.crt`):

```shell
pip install -i https://test.pypi.org/simple/ triton-ascend==3.2.0.dev2025103116
```

**f. Install jax-triton**

```shell
git clone https://github.com/jax-ml/jax-triton.git
cd jax-triton
git checkout tags/v0.2.0
```

In `pyproject.toml`, change `triton>=3.1` to `triton_ascend>=3.1`, then:

```shell
pip install -e .
cd ..
```

**g. tcmalloc (optional performance; `run.sh` may set `LD_PRELOAD`)**

Defaults in upstream scripts target **openEuler**. On **Ubuntu**, adjust `LD_PRELOAD` in `run.sh` and install tcmalloc accordingly.

- **openEuler** (example):

```shell
mkdir gperftools && cd gperftools
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz --no-check-certificate
tar -zvxf gperftools-2.16.tar.gz && cd gperftools-2.16
./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
make && make install
echo '/usr/local/lib/lib/' >> /etc/ld.so.conf
ldconfig
export LD_PRELOAD=/usr/local/lib/libtcmalloc.so.4
```

- **Ubuntu**: ensure `autoconf` and `libtool` are installed; build **libunwind** then **gperftools** per README (clone libunwind, `autoreconf -i`, `./configure --prefix=/usr/local`, `make install`; then gperftools 2.16 with same `--prefix=/usr/local/lib --with-tcmalloc-pagesize=64`). Set e.g. `export LD_PRELOAD="$LD_PRELOAD:/usr/local/lib/lib/libtcmalloc.so"`.

#### Pull model code

**a. AlphaFold 3 (sibling of `torchfold` recommended)**

Apply for and download weights **`af3.bin`** from the official AlphaFold 3 release, then place under `alphafold3/src/alphafold3/model/`.

```shell
git clone https://github.com/google-deepmind/alphafold3.git && cd alphafold3
git checkout a14376d249099b3e64ae8010290d4e20720d2698
git apply ../torchfold/alphafold3.patch
pip install -r dev-requirements.txt
pip install . --no-deps --verbose
build_data
cd ..
```

**b. TorchFold tree**

```shell
cd torchfold
pip install einops
pip install numpy==1.26.4
mkdir output
```

#### Dataset preparation

Download MSA / template and other databases per the [AlphaFold 3 model README](https://github.com/google-deepmind/alphafold3); point `DB_DIR` (or equivalent) in `scripts/env.sh` to those paths.

#### Inference entrypoint

Edit paths in `scripts/env.sh`, then from the `torchfold` directory:

```shell
bash run.sh
```

Pass the input JSON name if your `run.sh` expects it (e.g. `bash run.sh input_name.json`).

#### Environment Requirements (hardware / disk)

| Requirement | Specification |
| ----------- | ------------- |
| Hardware | Huawei Ascend NPU |
| Memory | Sufficient host and device memory for AlphaFold3-style inference |
| Disk Space | 50GB+ for databases and weights (see README and AlphaFold3 documentation) |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | `git clone https://gitcode.com/AI4Science/Mingchenchen.git` |
| 2 | Match **Component versions**; complete **Environment setup** (a–g): conda, hmmer, torch/torch_npu, DrivingSDK (`ENABLE_ONNX=False`), triton-ascend, jax-triton (`pyproject.toml`), tcmalloc if used |
| 3 | **Pull model code**: AlphaFold3 at pinned commit + `alphafold3.patch`, `build_data`, place `af3.bin`; then `torchfold` deps and `mkdir output` |
| 4 | **Dataset preparation** per AlphaFold3 README; set paths in `scripts/env.sh` |
| 5 | `bash run.sh` (and JSON argument if required by your `run.sh`) |

**Optional NPU availability check:**

```bash
python -c "import torch; import torch_npu; print(torch.__version__); print(torch_npu.npu.is_available())"
```

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type | Description |
| --------------- | ----------- |
| Functional Limitations | Requires AlphaFold 3 model weights; obtain and place them per the official README |
| Performance Limitations | Heavier than the reference JAX implementation on comparable hardware |
| Scale Limitations | Input sequence length is limited by available device memory; typical proteins under ~2000 residues |
| Input Format | Must use AlphaFold3 JSON dialect; limited validation of input format |

#### Notes

- **Note 1**: Model weights must be obtained separately — either from the original AlphaFold 3 release or from the TorchFold checkpoint releases
- **Note 2**: Follow the Mingchenchen repository README for Ascend-specific configuration
- **Note 3**: The model requires substantial computational resources; plan capacity per the official README
- **Note 4**: Database files (MSA, templates) are required for full inference; without them, predictions will be less accurate

---

### 4. Model Invocation Guide

#### Model Initialization

| Item | Example / value |
| ---- | ---------------- |
| Checkpoint | TorchFold PyTorch checkpoint or AlphaFold 3 JAX weights |
| Model class | `AlphaFold3` from `torchfold.alphafold3` |

#### Running inference (recommended path)

**Shell script:**

```bash
# Set paths in scripts/env.sh (JSON, output, DB, checkpoint, etc.) per README
cd /path/to/Mingchenchen/torchfold
bash run.sh
# If your run.sh expects a filename:
# bash run.sh input_name.json
```

**Python API:**

```python
import torch
import pytree
from torchfold.alphafold3 import AlphaFold3
from torchfold.params import import_jax_weights_
from alphafold3.common import folding_input
from alphafold3.data import featurisation
from alphafold3.constants import chemical_components

# Initialize model
model = AlphaFold3(num_samples=5)
model.eval()

# Load JAX weights
import_jax_weights_(model, model_dir_path)

# Set device (Ascend NPU when available)
device = torch.device(
    'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'
)
model = model.to(device=device)

# Load and featurize input
json_path = "/path/to/input.json"
fold_input = folding_input.load_fold_inputs_from_path(json_path)
ccd = chemical_components.cached_ccd()
featurised_examples = featurisation.featurise_input(
    fold_input=fold_input, buckets=None, ccd=ccd, verbose=True
)

# Run inference
with torch.inference_mode():
    example = pytree.tree_map(torch.from_numpy, featurised_examples[0])
    example = pytree.tree_map_only(torch.Tensor, lambda x: x.to(device=device), example)
    result = model(example)
```

#### Result Post-processing

- Output directory specified by `OUTPUT_DIR` environment variable
- Results include predicted structure files (CIF format by default)
- Multiple samples can be generated with `num_diffusion_samples` parameter
- Results include confidence scores (pLDDT) for each residue

---

## Reference Resources

- **GitCode repository**: https://gitcode.com/AI4Science/Mingchenchen
- **Official README**: https://gitcode.com/AI4Science/Mingchenchen/blob/main/README.md
- **Additional reference (upstream)**: https://github.com/Mingchenchen/TorchFold
- **Original AlphaFold 3**: https://github.com/google-deepmind/alphafold3
- **AlphaFold 3 Paper**: https://www.nature.com/articles/s41586-021-03819-2