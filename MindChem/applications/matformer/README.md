ENFLISH | [简体中文](README_CN.md)

# MatFormer: Predicting Formation Energy per Atom for Crystalline Materials

## Model Overview

### Background

> [MatFormer](https://arxiv.org/abs/2209.11807) is a state-of-the-art (SOTA) model based on **Graph Neural Networks (GNNs)** and the **Transformer** architecture, specifically designed for predicting various properties of crystalline materials.  
> The model can handle the periodic graph structures of crystals, capturing both local and global structural information while preserving periodicity invariance. Compared to traditional models such as CGCNN, SchNet, and MEGNet, MatFormer demonstrates superior performance in predicting material properties including formation energy, bandgap, and lattice constants.

### Model Architecture

![alt text](image.png)

The overall workflow of MatFormer is as follows:

1. **Input Feature Extraction**:
   - Atomic features \( a_i \) are embedded via a CGCNN embedding layer to obtain initial node representations \( f^*_i \).
   - Interatomic distances \( d_{ij}^h \) are transformed into Gaussian radial basis features \( e_{ij}^h \) using an RBF kernel.
   - The edge features \( e_{ij}^h \) are further processed by a Linear layer followed by a Softplus activation before being fed into the network.

2. **Stacked MatFormer Layers**:
   - Multiple MatFormer layers are stacked sequentially, each updating node and edge representations.
   - Each layer takes the current node states \( f^*_i \) and edge features \( e_{ij}^h \) as input.

3. **Readout Layer**:
   - Final node representations are aggregated via Average Pooling.
   - A final prediction is produced through a Linear → SiLU → Linear module.

---

Each MatFormer Layer consists of the following core components:

### 1. Attention Mechanism

- **Query (Q), Key (K), Value (V)**:
    - Derived from \( f^*_i \) and \( f^*_j \) via distinct linear transformations \( \text{LN}_Q, \text{LN}_K, \text{LN}_V \).
    - Edge features \( e_{ij}^h \) are also incorporated after transformation by \( \text{LN}_E \).

- **Multi-head Attention**:
    - The figure illustrates two attention heads (Head1, Head2); the number of heads is configurable.
    - Hadamard (element-wise) products are used to fuse node and edge features.

- **Attention Weight Computation**:
    - Q and K pass through LayerNorm, Sigmoid, and Hadamard operations to generate attention weights.
    - The final attention output is:
    $$
    \sum_{j \in N_i} \sum_h [\text{Head1}, \text{Head2}]
    $$

### 2. Aggregation

- Neighbor information is aggregated into a message \( m_i \).
- This involves:
    - Concatenating neighbor representations.
    - Applying linear transformations.
    - Processing through multiple attention heads (e.g., Head1, Head2).

### 3. Update

- The current node state \( f^*_i \) is combined with the aggregated message \( m_i \):
  $$
  f'_i = f^*_i \oplus m_i
  $$
  where \( \oplus \) denotes a residual connection or concatenation.
- The message \( m_i \) is refined through LayerNorm, Linear layers, Hadamard products, and an activation function \( \sigma(\text{BN}) \).

### 4. Normalization and Nonlinear Activation

- LayerNorm and Sigmoid are employed to stabilize training.
- Hadamard products facilitate information fusion from different pathways.

---

## Dataset

> Download `jdft_3d-12-12-2022.json` from <https://figshare.com/articles/dataset/jdft_3d-7-7-2018_json/6815699> into the current directory without renaming the file.

### Basic Information

- **Dataset Name**: `jdft_3d-12-12-2022.json`
- **Source**: [JARVIS-DFT](https://jarvis.nist.gov/) (Joint Automated Repository for Various Integrated Simulations – Density Functional Theory)
- **Size**: **75,993** 3D bulk crystal structures
- **Format**: JSON
- **Material Identifier**: Unique `jid` (e.g., `JVASP-90856`)

---

### Dataset Overview

This dataset contains **3D bulk crystalline materials** with properties computed via **Density Functional Theory (DFT)**, suitable for materials discovery, property prediction, and machine learning modeling.

---

### Key Fields

| Field | Type | Description |
|------|------|-------------|
| `jid` | str | Unique JARVIS material ID (e.g., `JVASP-90856`) |
| `formula` | str | Chemical formula (e.g., `TiCuSiAs`) |
| `spg_number` / `spg_symbol` | int / str | Space group number and symbol (e.g., 129, `P4/nmm`) |
| `formation_energy_peratom` | float | Formation energy per atom (eV/atom); more negative values indicate greater stability |
| `optb88vdw_bandgap` | float | Bandgap computed with the OptB88vdW functional (eV) |
| `mbj_bandgap`, `hse_gap` | float | Bandgaps from mBJ or HSE06 functionals (available for some materials) |
| `atoms` | dict | **Core structural data**, including:<br>• `lattice_mat`: 3×3 lattice matrix<br>• `coords`: atomic coordinates<br>• `elements`: list of elements<br>• `cartesian`: whether coordinates are Cartesian (bool) |
| `density` | float | Material density (g/cm³) |
| `ehull` | float | Energy above convex hull (eV/atom); < 0.1 eV/atom is considered stable |
| `func` | str | DFT functional used (e.g., `OptB88vdW`) |
| `dimensionality` | str | Material dimensionality (all entries are `3D-bulk`) |
| `crys` | str | Crystal system (e.g., `tetragonal`, `cubic`) |
| `nat` | int | Total number of atoms |
| `reference` | str | Corresponding Materials Project ID (e.g., `mp-1080455`) |

---

## Environment Requirements

> 1. Install `mindspore`  
> 2. Install `mindscience`

## Quick Start

### Training Method 1

```bash
pip install -r requirements.txt
python train.py
```

Training hyperparameters are defined in `config.yaml`. Modify this file to adjust training settings.

To perform prediction, specify the model checkpoint path in `config.yaml` under predictor.checkpoint_path:

```bash
python predict.py
```

All configurations are managed via `config.yaml`

### Training Method 2: Jupyter Notebook

You can run the training and evaluation code step-by-step using the provided Jupyter Notebooks (available in both [Chinese](matformer_application.ipynb) and [English](matformer_application_EN.ipynb) versions).

## Results

The figure below shows the formation energy predictions from a fully trained MatFormer model. The predicted values closely match the ground truth, indicating low prediction error.
![alt text](image-1.png)

### Training Logs

Log output from `train.py`:

```log
INFO:root:The model you built has 2786689 parameters.
INFO:root:Starting new training process
INFO:root:Start to initialise train loader
INFO:root:Start to initialise eval loader
INFO:root:+++++++++++++++ start traning +++++++++++++++++++++
INFO:root:==============================step: 0 ,epoch: 0
INFO:root:learning rate: 4e-05
INFO:root:train mse loss: 0.8999285
INFO:root:is_finite: True
INFO:root:training time: 51.66963744163513
.
.
.
INFO:root:step:117, epoch: 499
INFO:root:validation mse loss: 0.004059551
INFO:root:validation mae loss: 0.034488887
INFO:root:validation time: 0.041112422943115234
INFO:root:epoch 499 running time: 137.772692
INFO:root:epoch 499 average train mse loss: 0.0003474082
INFO:root:epoch 499 average validation mse loss: 0.00414170
INFO:root:epoch 499 average validation mae loss: 0.03259226
```

Log output from Jupyter Notebook:

```log
Model trainable parameters: %s 2786689
Starting new training process
.Saved best model at epoch %d, MSE: %.6f 0 0.0991247
Epoch 0 | Train MSE: 0.152263 | Val MSE: 0.099125 | Val MAE: 0.211994
```