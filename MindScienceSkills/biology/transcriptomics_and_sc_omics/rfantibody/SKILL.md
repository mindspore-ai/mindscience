---
name: rfantibody
description: rfantibody is a structure-based de novo antibody and nanobody design pipeline (RFdiffusion → ProteinMPNN → antibody-finetuned RF2), typically run in Docker with Poetry. PyTorch Geometric (PyG) is supported as the graph backend with Ascend NPU optimizations. Use when designing CDR loops and docks against a target epitope.
license: MIT
metadata:
    skill-author: MindSpore Science Team
    hardware-requirements: Ascend
---

# RFantibody

## Overview

RFantibody is a pipeline for structure-based _de novo_ antibody and nanobody design. It combines three methods:

1. **Protein backbone design** — antibody-finetuned [RFdiffusion](https://www.nature.com/articles/s41586-023-06415-8)
2. **Sequence design** — [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187)
3. **_In silico_ filtering** — antibody-finetuned [RoseTTAFold2](https://www.biorxiv.org/content/10.1101/2023.05.24.542179v1)

Pipeline details: [RFantibody preprint](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1).

### PyG graph backend

RFantibody supports **PyTorch Geometric (PyG)** as a graph backend, including Ascend **910B**-oriented optimizations, a DGL-compatible interface, optional AscendC-accelerated graph ops, automatic mixed precision, and large speedups versus DGL-on-CPU in reported setups. See repository **`BACKEND_USAGE.md`** for full options.

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

- **Scenario 1**: Generate antibody–target docks and CDR backbones with hotspot-conditioned RFdiffusion.
- **Scenario 2**: Assign CDR sequences to docked frameworks with ProteinMPNN.
- **Scenario 3**: Score and filter designs with antibody-finetuned RF2 (pAE, RMSD vs design, optional Rosetta filters).

---

## Requirements

### Docker

RFantibody is intended to run in **Docker** (reproducible environment, simplified install). Install [Docker Engine](https://docs.docker.com/engine/install/) on the host. Verify:

```bash
which docker
```

### Hardware acceleration

| Backend | Check |
| ------- | ----- |
| **NVIDIA CUDA** | `nvidia-smi` |
| **Huawei Ascend** | `npu-smi info` |
| **CPU** | Supported; slower |

Device selection priority (when using auto): **NPU > CUDA > CPU**.

---

## Downloading weights

From the RFantibody repository root on the host:

```bash
bash include/download_weights.sh
```

Weights are written under `weights/` at the repository root.

---

## Installation

### Host: Docker group

```bash
sudo usermod -aG docker $USER
```

Log out and back in so group membership applies.

#### Verified Ascend stack (NPU)

| Component | Version |
| --------- | ------- |
| HDK | 25.0.RC1 |
| CANN | 8.3.RC1 |
| Python | 3.9 |
| torch | 2.1.0 |
| torch-npu | 2.1.0.post14 |

**Note:** Install a `torch-npu` build that matches your CANN; pair HDK and firmware with that CANN release per Huawei Ascend documentation.

### Build image

```bash
docker build -t rfantibody .
```

### Start container (example — GPU)

```bash
docker run --name rfantibody --gpus all -v .:/home --memory 10g -it rfantibody
```

The workspace is mounted at **`/home`** inside the container (mirrors the directory from which you ran `docker run`). On Ascend-only hosts, use the image and device flags described in **`BACKEND_USAGE.md`** and your cluster docs instead of `--gpus all` where appropriate.

### Python environment inside the container

```bash
bash /home/include/setup.sh
```

This prepares DGL-related artifacts (for fallback paths), builds the environment with **Poetry**, and builds the **[USalign](https://github.com/pylelab/USalign)** executable.

---

## Backend configuration

### Graph backend

**PyG (recommended for NPU / performance):**

```bash
export GRAPH_BACKEND=pyg
export DEVICE=auto   # or npu:0, cuda:0, cpu
```

**DGL (fallback):**

```bash
export GRAPH_BACKEND=dgl
export DEVICE=cpu    # typical for DGL path
```

### Verify configuration

```python
from rfantibody.rf2.network.backend_config import print_backend_info
print_backend_info()
```

Example fields: `device`, `type`, `graph_backend`, `auto_mixed_precision`.

### NPU-oriented options (optional)

```bash
export GRAPH_BACKEND=pyg
export DEVICE=npu:0
export USE_ASCENDC_EDGE_SOFTMAX=1
export NPU_COMPILE_MODE=jit
```

Details: **`BACKEND_USAGE.md`**.

---

## Data formats and inputs

### HLT file format

Pipeline steps exchange **HLT** files: PDB with conventions:

- **H** = heavy chain, **L** = light chain, **T** = target (all target chains use `T`)
- Chain order: **Heavy → Light → Target**
- **REMARK** lines at the end give **1-indexed absolute** (whole-structure) residue indices for each CDR, e.g.:

```
REMARK PDBinfo-LABEL:   32 H1
REMARK PDBinfo-LABEL:   52 H2
```

### Input preparation (Chothia → HLT)

Inside the container:

```bash
poetry run python /home/scripts/util/chothia_to_HLT.py -inpdb mychothia.pdb -outpdb myHLT.pdb
```

Expects **Chothia-annotated** PDB; [SabDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab) is a common source.

Example frameworks from the paper:

- Nanobody: `RFantibody/scripts/examples/example_inputs/h-NbBCII10.pdb`
- ScFv: `RFantibody/scripts/examples/example_inputs/hu-4D5-8_Fv.pdb`

---

## Pipeline usage (inside container)

Paths below use **`/home`** as the repo root inside Docker.

### 1. RFdiffusion (backbone / dock generation)

```bash
poetry run python /home/src/rfantibody/scripts/rfdiffusion_inference.py \
    --config-name antibody \
    antibody.target_pdb=/home/scripts/examples/example_inputs/rsv_site3.pdb \
    antibody.framework_pdb=/home/scripts/examples/example_inputs/hu-4D5-8_Fv.pdb \
    inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt \
    'ppi.hotspot_res=[T305,T456]' \
    'antibody.design_loops=[L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13]' \
    inference.num_designs=20 \
    inference.output_prefix=/home/scripts/examples/example_outputs/ab_des
```

| Key | Role |
| --- | ---- |
| `antibody.target_pdb` | Target structure (often cropped; see practical notes) |
| `antibody.framework_pdb` | HLT framework; only annotated loops are designed |
| `inference.ckpt_override_path` | RFdiffusion antibody weights |
| `ppi.hotspot_res` | Epitope hotspots (RFdiffusion-style residue list) |
| `antibody.design_loops` | Per-CDR allowed length ranges; omitted loops fixed; dict entry without range fixes length from framework |
| `inference.num_designs` | Number of designs |
| `inference.output_prefix` | Output PDB prefix |

Example script:

```bash
bash /home/scripts/examples/rfdiffusion/antibody_pdbdesign.sh
```

### 2. ProteinMPNN (CDR sequences)

Directory of HLT PDBs:

```bash
poetry run python /home/scripts/proteinmpnn_interface_design.py \
    -pdbdir /path/to/inputdir \
    -outpdbdir /path/to/outputdir
```

```bash
poetry run python /home/scripts/proteinmpnn_interface_design.py --help
```

Example:

```bash
bash /home/scripts/examples/proteinmpnn/ab_pdb_example.sh
```

### 3. RF2 (structure prediction / filtering)

```bash
poetry run python /home/scripts/rf2_predict.py \
    input.pdb_dir=/path/to/inputdir \
    output.pdb_dir=/path/to/outputdir
```

Defaults include **10** recycling iterations and **10%** of hotspots exposed to the model (tunable as more campaign data appears).

Example:

```bash
bash /home/scripts/examples/rf2/ab_pdb_example.sh
```

---

## Practical considerations (summary)

- **Target site**: Prefer sites with several hydrophobic residues; charged polar sites, glycan-proximal sites, and unstructured loops are harder (see RFdiffusion literature for peptide/loop strategies).
- **Nanobody docks**: Side-on docks reflect training; use an antibody framework if you want antibody-like geometry.
- **Truncation**: Runtime scales roughly **O(N²)** in system size; crop target while preserving secondary structure and ~10 Å context around the epitope (e.g. PyMOL).
- **Hotspots**: RFantibody is sensitive to hotspot choice; pilot runs before large campaigns.
- **Scale**: Some campaigns found binders in ~95 designs; often **~10k** designs may be needed without a strong filter.
- **CDR length ranges**: Example ranges in repo examples mirror natural length distributions; lengthen **H3** when targeting deep hydrophobic pockets.
- **Minimal filtering (starting point)**: RF2 **pAE < 10**; **RMSD (design vs RF2) < 2 Å**; optionally Rosetta **ddG < -20**. Filter quality remains a main limitation; AF-class predictors are under evaluation as alternatives to RF2.

---

## Quiver files

For large campaigns, [Quiver](https://github.com/nrbennet/quiver) packs many designs and scores in one file (lighter on filesystems than thousands of PDBs). CLI tools mirror ideas from [silent_tools](https://github.com/bcov77/silent_tools).

Common commands:

```bash
qvfrompdbs *.pdb > my.qv
qvls my.qv
qvls my.qv | wc -l
qvextract my.qv
qvls my.qv | head -n 10 | qvextractspecific my.qv
qvls my.qv | shuf | head -n 10 | qvextractspecific my.qv
qvextractspecific my.qv name_of_pdb_0001
qvscorefile my.qv
cat 1.qv 2.qv 3.qv > my.qv
qvls my.qv | qvrename my.qv > uniq.qv
qvsplit my.qv 100
```

### Pipeline integration

- **RFdiffusion**: append `inference.quiver=/path/to/myoutput.qv`
- **ProteinMPNN**: `-inquiver` / `-outquiver`
- **RF2**: `input.quiver=` / `output.quiver=`

---

## Advanced: PyG and NPU

- Unified backend API (DGL-compatible surface), auto device order **NPU > CUDA > CPU**, mixed precision.
- Example:

```python
import os
os.environ["GRAPH_BACKEND"] = "pyg"
os.environ["DEVICE"] = "npu:0"
from rfantibody.rf2.network.backend_config import config
print(f"Using {config.graph_backend} on {config.device}")
```

- Reported NPU-optimized ops include `edge_softmax`, `gspmm`, `gsddmm`, `e_dot_v`, `copy_e_sum`. Prefer feature dims aligned to **8** on NPU; larger batches and mixed precision help throughput.

---

## Hardware and disk

| Resource | Notes |
| -------- | ----- |
| RAM | Container example uses **10g** memory flag; scale for full campaigns |
| GPU / NPU | Match backend (`DEVICE`, Docker runtime) |
| Disk | Weights plus PDB/Quiver outputs; large S3-style campaigns need substantial storage |

---

## End-to-end checklist

- [ ] Install Docker; add user to `docker` group if needed
- [ ] Clone RFantibody; `bash include/download_weights.sh`
- [ ] `docker build -t rfantibody .` and `docker run ... -v .:/home ...`
- [ ] `bash /home/include/setup.sh` inside container
- [ ] Set `GRAPH_BACKEND` / `DEVICE` (and optional NPU flags); run `print_backend_info()`
- [ ] Prepare **Chothia → HLT** or use example frameworks/targets
- [ ] Run RFdiffusion → ProteinMPNN → RF2 (or example shell scripts under `/home/scripts/examples/`)
- [ ] Optional: Quiver I/O for scale; filter outputs (pAE, RMSD, ddG)
- [ ] Read **`BACKEND_USAGE.md`** and **`public_address_statement.md`** (if present) for URLs and policy

---

## Reference resources

- **RFantibody (RosettaCommons)**: https://github.com/RosettaCommons/RFantibody
- **RFantibody preprint**: https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1
- **RFdiffusion**: https://www.nature.com/articles/s41586-023-06415-8
- **ProteinMPNN**: https://www.science.org/doi/10.1126/science.add2187
- **RoseTTAFold2 (antibody finetune basis)**: https://www.biorxiv.org/content/10.1101/2023.05.24.542179v1
- **Quiver**: https://github.com/nrbennet/quiver
- **silent_tools**: https://github.com/bcov77/silent_tools
- **SabDab**: https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab
- **Docker**: https://docs.docker.com/engine/install/

License: **MIT** — see **`LICENSE`**. Acknowledgments for RoseTTAFold, RFdiffusion, and ProteinMPNN are included in the repository’s documentation files.
