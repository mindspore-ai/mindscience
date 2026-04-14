---
name: ligandmpnn
description: ligandmpnn (Ligand Message Passing Neural Network) is a deep learning model for atomic context-conditioned protein sequence design. It incorporates ligand and other non-protein atoms (such as DNA, RNA, or small molecules) as context for protein sequence design. Use this model when you need to design novel amino acid sequences for protein backbones with ligand awareness.
license: MIT
metadata:
    skill-author: MindSpore Science Team

---

# LigandMPNN

## Overview

LigandMPNN is an extension of ProteinMPNN that enables atomic context-conditioned protein sequence design. The model uses a message passing neural network architecture to generate amino acid sequences for given protein backbones while considering ligand atoms, cofactors, metals, DNA, or RNA as contextual information.

This skill provides inference capabilities adapted for Ascend NPU, enabling users to run protein sequence design tasks on Huawei Ascend NPUs.

---

## When to Use

- **Ligand-aware protein design**: Design protein sequences that consider ligand binding sites, cofactors, or metal ions
- **Enzyme engineering**: Generate sequences for enzymes with known ligand/substrate complexes
- **Drug target design**: Design protein sequences around small molecule drug targets
- **Multi-chain complex design**: Design sequences for protein complexes with nucleic acids or small molecules
- **Homooligomer design**: Generate sequences for symmetric oligomeric protein assemblies

---

## How It Works

### 1. Dataset Acquisition and Processing

#### Dataset Requirements

| Requirement | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| Data Format | PDB files (.pdb) containing protein structures with optional ligand atoms |
| Data Size   | Single PDB files or directories containing multiple PDB files |
| Data Source | Bundled smoke-test inputs in **`LigandMPNN/inputs`** (inner repo); RCSB PDB, AlphaFold DB, or custom PDBs for your own runs |

#### Data Acquisition Methods

1. **RCSB PDB Download**: Download structures from https://www.rcsb.org/
2. **AlphaFold DB**: Use predicted structures from https://alphafold.ebi.ac.uk/
3. **Custom PDB Files**: Prepare PDB files with protein backbone and ligand atoms

#### Data Preprocessing

- Ensure PDB files contain ATOM records for protein backbone (N, CA, C atoms)
- Include HETATM records for ligand atoms if ligand context is needed
- Remove water molecules (HOH) unless specifically required
- Verify chain identifiers and residue numbering are correct

---

### 2. Environment Configuration and Dependencies

#### Verified Ascend stack (reference)

| Component | Version     |
| --------- | ----------- |
| HDK       | 24.1.RC3    |
| CANN      | 8.2.RC1     |
| Python    | 3.11        |
| torch     | 2.3.1       |
| torch-npu | 2.3.1.post4 |

Install **HDK** and **CANN** on the host per Huawei documentation before the Python steps. Match **torch** / **torch-npu** to your CANN build; the versions above are the tested combination for this workflow.

#### Clone LifeScience and prepare patched upstream code

From a working directory of your choice:

```bash
git clone https://gitcode.com/AI4Science/LifeScience.git
cd LifeScience/PyTorch/LigandMPNN

git clone https://github.com/dauparas/LigandMPNN.git && cd LigandMPNN
git checkout 26ec57ac976ade5379920dbd43c7f97a91cf82de
git apply ../patch/LigandMPNN.patch
```

After this, **`run_examples.sh`**, **`get_model_params.sh`**, and **`requirements.txt`** live under the **inner** `LigandMPNN/` directory (the cloned upstream tree with the patch applied). The Ascend patch file is `LifeScience/PyTorch/LigandMPNN/patch/LigandMPNN.patch`.

#### Conda environment and Python dependencies

```bash
conda create --name ligandmpnn python=3.11
conda activate ligandmpnn
cd /path/to/LifeScience/PyTorch/LigandMPNN/LigandMPNN
pip install -r requirements.txt
```

Install **torch** and **torch-npu** at the versions in the table (or as required by your CANN release) if `requirements.txt` does not already match; avoid CUDA-only PyTorch wheels on Ascend.

#### Environment Requirements (hardware / disk)

| Requirement | Specification                                  |
| ----------- | ---------------------------------------------- |
| Hardware    | Ascend NPU (910 series recommended)            |
| Memory      | At least 16GB RAM recommended                  |
| Disk Space  | At least 5GB for model checkpoints and outputs |

#### End-to-end checklist

| Step | Action                                                       |
| ---- | ------------------------------------------------------------ |
| 1    | Clone **LifeScience**, enter `PyTorch/LigandMPNN`, clone **dauparas/LigandMPNN**, checkout commit `26ec57ac976ade5379920dbd43c7f97a91cf82de`, apply `../patch/LigandMPNN.patch`. |
| 2    | Install HDK / CANN; create conda env `ligandmpnn` (Python 3.11); `pip install -r requirements.txt` from the **inner** `LigandMPNN/` directory. |
| 3    | **Data**: use bundled examples under **`LigandMPNN/inputs`** for a first run; add your own PDBs as needed. |
| 4    | **Weights**: from the **inner** `LigandMPNN/` directory: `bash get_model_params.sh "./model_params"`. |
| 5    | **Inference**: from the **inner** `LigandMPNN/` directory: `bash run_examples.sh`. |

Use Linux or WSL (or Git Bash on Windows) so `bash` is available.

Optional NPU check:

```bash
python -c "import torch; print(torch.__version__); print(getattr(torch, 'npu', None) and torch.npu.is_available())"
```

---

### 3. Usage Limitations and Notes

#### Model Limitations

| Limitation Type         | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| Functional Limitations  | Requires protein backbone coordinates (N, CA, C atoms); cannot design de novo structures |
| Performance Limitations | Inference time scales with number of residues; larger proteins take longer |
| Scale Limitations       | Recommended maximum ~2000 residues per design for typical hardware |
| Input Format            | Must be valid PDB format with proper atom naming             |

#### Notes

- **Note 1**: LigandMPNN requires ligand atoms to be present in the PDB file when using `ligand_mpnn` model type. If no ligand is present, use standard `protein_mpnn` model type.
- **Note 2**: For NPU inference, install the **HDK** / **CANN** stack (e.g. HDK 24.1.RC3 and CANN 8.2.RC1 with torch 2.3.1 + torch-npu 2.3.1.post4 as in §2) and verify the NPU is visible to PyTorch.
- **Note 3**: Confidence scores range from 0.0 (low confidence) to 1.0 (high confidence). Values > 0.8 indicate high-confidence predictions.
- **Note 4**: Hyperparameters such as batch count and sampling temperature are configured inside **`run_examples.sh`** or the Python modules it calls—see comments in that script and any README under `LifeScience/PyTorch/LigandMPNN`.

---

### 4. Model Invocation Guide

#### Model Initialization

Common checkpoint filenames for LigandMPNN / ProteinMPNN weights (exact paths are set in **`run_examples.sh`** or the README):

| Model Type  | Example checkpoint        | Noise Level |
| ----------- | ------------------------- | ----------- |
| LigandMPNN  | ligandmpnn_v_32_005_25.pt | 0.05Å       |
| LigandMPNN  | ligandmpnn_v_32_010_25.pt | 0.10Å       |
| LigandMPNN  | ligandmpnn_v_32_020_25.pt | 0.20Å       |
| LigandMPNN  | ligandmpnn_v_32_030_25.pt | 0.30Å       |
| ProteinMPNN | proteinmpnn_v_48_002.pt   | 0.02Å       |

#### Running LifeScience examples

**Shell (recommended):** run from the **patched inner** `LigandMPNN` repository (same directory as `get_model_params.sh` and `requirements.txt`).

```bash
cd /path/to/LifeScience/PyTorch/LigandMPNN/LigandMPNN
bash run_examples.sh
```

**Python helper** (`scripts/inference_helper.py` in this skill): set `ligandmpnn_repo_path` to that **inner** `LigandMPNN` directory. Add the skill’s `…/ligandmpnn` folder to `PYTHONPATH` if you import `LigandMPNNRunner` from this repo.

```python
from scripts.inference_helper import LigandMPNNRunner

runner = LigandMPNNRunner(
    ligandmpnn_repo_path="/path/to/LifeScience/PyTorch/LigandMPNN/LigandMPNN"
)
runner.run_examples()  # if run_examples.sh forwards "$@", pass extra_args=["foo", "bar"]
```

#### Result Post-processing

Output paths and file types follow **`run_examples.sh`** and the README (often sequence FASTA, structure PDB, optional statistics or side-chain packing). Before running, read `out_folder` or equivalent settings in the script.

---

## Reference Resources

- **Ascend / LifeScience (this skill)**: https://gitcode.com/AI4Science/LifeScience/tree/main/PyTorch/LigandMPNN
- **ProteinMPNN Paper**: https://www.science.org/doi/10.1126/science.add2187
- **LigandMPNN Paper**: https://www.biorxiv.org/content/10.1101/2023.12.22.573103v1
- **RCSB PDB**: https://www.rcsb.org/
- **AlphaFold DB**: https://alphafold.ebi.ac.uk/