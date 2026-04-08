---
name: diffdock-model
description: diffdock-model is a deep learning model for molecular docking that predicts the 3D structure of protein-ligand complexes using diffusion-based generative modeling. Use this model when you need to predict binding poses of small molecule ligands to protein targets on Ascend NPU with the LifeScience patched tree.
license: MIT
metadata:
    skill-author: MindSpore Science Team
---

# DiffDock

## Overview

DiffDock is a molecular docking model that uses diffusion-based generative modeling to predict 3D binding poses of small molecule ligands to protein targets in a blind docking setting. DiffDock-L improves performance and generalization over the original release.

---

## When to Use

- **Scenario 1**: Small-molecule docking — predict poses of drug-like ligands against a protein structure.
- **Scenario 2**: Virtual screening follow-up — generate 3D poses for hits from screening.
- **Scenario 3**: Interaction analysis — explore how a ligand may bind in the protein frame.

---

## How It Works

### 1. Dataset acquisition and processing

#### Dataset requirements

| Requirement | Description |
| ------------- | ----------- |
| Data format | Protein: PDB; ligand: SDF, MOL2, or SMILES (RDKit-parseable) |
| Batch | CSV with columns such as `complex_name`, `protein_path`, `ligand_description`, `protein_sequence` (optional) |
| Source | RCSB PDB, PubChem, or custom files |

#### Dataset preparation (bundled examples)

Use the **built-in examples** under `DiffDock/data` to validate the model. For other complexes, download or prepare PDB and ligand files yourself and point inference to those paths.

#### Preprocessing (custom data)

- Provide a full protein PDB (DiffDock uses the whole structure).
- Ensure the ligand is in a format RDKit can read.
- For batches, build a CSV as required by `inference` / your script wrapper.

---

### 2. Environment configuration and dependencies

#### Component versions

```shell
hdk: 25.0.RC1
cann: 8.3.RC1
python: 3.9
torch: 2.1.0
torch-npu: 2.1.0.post14
```

#### Clone model code

```bash
git clone https://gitcode.com/AI4Science/LifeScience.git
cd LifeScience/PyTorch/DiffDock
git clone https://github.com/gcorso/DiffDock.git && cd DiffDock
git checkout 85c49b60d3e0b0182a59ee43a34a6d7036981284
git apply ../patch/DiffDock.patch
```

#### Environment setup

1. **Create conda environment**

```bash
conda create --name diffdock python=3.9
conda activate diffdock
```

2. **Install PyTorch for Ascend**

Install `torch==2.1.0` and `torch_npu==2.1.0.post14` using the wheels or index matching your **CANN** installation.

3. **Install Python requirements**

```bash
pip install -r requirements.txt
```

#### Environment requirements (hardware / disk)

| Requirement | Specification |
| ----------- | -------------- |
| Hardware | Huawei Ascend NPU |
| Memory | 16GB+ host RAM recommended; device memory depends on workload |
| Disk | ~5GB+ for weights and dependencies |

#### End-to-end checklist

| Step | Action |
| ---- | ------ |
| 1 | Clone LifeScience, enter `PyTorch/DiffDock`, clone upstream DiffDock, checkout commit, apply `DiffDock.patch` |
| 2 | Conda env Python 3.9; install `torch==2.1.0`, `torch_npu==2.1.0.post14`; `pip install -r requirements.txt` |
| 3 | Run bundled examples from `DiffDock/data` or your own data |
| 4 | Run `bash test/inference.sh` (first run can take a long time) |

**Optional NPU check:**

```bash
python -c "import torch; import torch_npu; print(torch.__version__); print(torch_npu.npu.is_available())"
```

---

### 3. Inference

The **first** inference run can be **slow** (e.g. SO(2)/SO(3) table preparation and setup); wait for it to finish.

**Recommended entrypoint:**

```bash
bash test/inference.sh
```

Run this from the **patched `DiffDock` repository root** (inside the conda environment).

#### Alternative: direct `inference` module

```bash
cd DiffDock
python -m inference \
  --config default_inference_args.yaml \
  --protein_path data/1a0q/1a0q_protein_processed.pdb \
  --ligand data/1a0q/1a0q_ligand.sdf \
  --out_dir results/output
```

**Ligand as SMILES:**

```bash
python -m inference \
  --config default_inference_args.yaml \
  --protein_path protein.pdb \
  --ligand "COc(cc1)ccc1C#N" \
  --out_dir results/output
```

**Batch CSV:**

```bash
python -m inference \
  --config default_inference_args.yaml \
  --protein_ligand_csv data/protein_ligand_example.csv \
  --out_dir results/user_predictions
```

**Gradio UI:**

```bash
python app/main.py
# http://localhost:7860
```

---

### 4. Usage limitations and notes

#### Model limitations

| Limitation | Description |
| ---------- | ----------- |
| Scope | Small-molecule ligands; not protein–protein or protein–nucleic acid docking |
| Performance | CPU is slow; use the Ascend stack above for supported deployments |
| Input | Valid protein PDB and RDKit-parseable ligand |

#### Notes

- Weights may be fetched automatically if missing (e.g. from project releases).
- Without a PDB, some workflows use sequence-driven structure prediction (e.g. ESMFold) as documented upstream.
- Confidence scores are roughly in **−3 … +1**; higher usually means higher model confidence.

---

### 5. Result post-processing

- **`--out_dir`**: e.g. `results/user_inference`
- Typical outputs: `rank1.sdf`, `rank{1-10}_confidence{score}.sdf`, optional visualization PDBs with `--save_visualisation`
- **Confidence**: `c > 0` higher confidence; `−1.5 < c < 0` moderate; `c < −1.5` low

---

## Reference resources

- **GitCode LifeScience (DiffDock)**: https://gitcode.com/AI4Science/LifeScience/tree/main/PyTorch/DiffDock
- **Upstream DiffDock**: https://github.com/gcorso/DiffDock
- **Paper**: DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking (Corso et al.)
- **Model weights**: https://github.com/gcorso/DiffDock/releases/latest/download/diffdock_models.zip
