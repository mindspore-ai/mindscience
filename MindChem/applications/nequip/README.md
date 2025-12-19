# NequIP

## Background

Paper link: [NequIP](https://arxiv.org/abs/2101.03164).
**NequIP (Neural Equivariant Interatomic Potential)** is a molecular potential prediction model based on **E(3)-Equivariant Graph Neural Networks**.
Its core idea is to ensure **physical consistency** — that the model’s predictions remain invariant under physical symmetries such as rotation, translation, and reflection.

Compared with traditional atomic potential models (e.g., Behler–Parrinello networks or handcrafted feature methods like SOAP), NequIP offers the following advantages:

- **Physical Consistency**: Model outputs are equivariant under 3D spatial transformations.
- **High Data Efficiency**: Achieves quantum-chemical accuracy with very limited training samples.
- **Generalizability & Scalability**: Applicable to a wide range of molecular and material systems.

## Model Implementation

### Hardware Requirements

- The training script `train.py` is currently verified on `Ascend` devices. Specify the device ID via `--device_target` and `--device_id` (see `train.py`).
- The inference script `predict.py` supports `Ascend` and `GPU` backends, switchable via `--device_target` (see `predict.py`).

### Version Dependencies

- Recommended: `MindSpore >= 2.7.0`.
- Install `MindScience` (which includes MindChem modules).

### Installation

- Install MindSpore: Refer to the official installation guide `https://www.mindspore.cn/install`
- Install MindScience: Refer to `https://atomgit.com/mindspore-lab/mindscience`

### Model Architecture

NequIP is built upon a **Message Passing Neural Network (MPNN)** framework and incorporates **Spherical Harmonic Tensor Representations** to achieve E(3)-equivariance in 3D space.

![alt text](images/architecture.png)

1. **Input Embedding Layer**
   Inputs atomic species (atomic numbers) and 3D coordinates.
   Each atom is mapped to an initial feature vector; coordinates are used to compute pairwise distances.

2. **Message Passing Layers**
   Capture local interactions between neighboring atoms.
   Messages are encoded as spherical tensors to maintain rotational equivariance.
   Feature update rule:

   $$
   h_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} \Phi\left(h_i^{(l)}, h_j^{(l)}, r_{ij}\right)
   $$

   where $r_{ij}$ denotes the relative position between atoms, and $\Phi\ $ is the equivariant linear transformation.

3. **Tensor Product Layers**
   Maintain E(3)-equivariance during feature propagation.
   Use tensor products between atomic features and spherical harmonics to encode directional dependencies.

4. **Readout Layer**
   Aggregate node features to output the **total molecular energy **$E$**.
   Atomic forces are derived from the energy gradient with respect to atomic positions:

   $$
   \mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}
   $$

### Dataset

The dataset used in this experiment is the **Uracil subset** of the **RMD17** dataset, located at: `dataset/RMD17/npz_data/rmd_uracil.npz`.

- **Dataset Name**: `rmd17_uracil.npz`
- **Source**: [Revised MD-17 (RMD-17)](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)
- **Target Molecule**: **Uracil**
- **File Format**: `.npz` (NumPy compressed format)
- **Atoms per Structure**: 24 (including hydrogens)

This dataset contains **molecular dynamics (MD)** trajectories of uracil, including **atomic configurations, energies, and forces** computed via high-accuracy quantum chemistry methods.
Compared to the original MD17, the RMD17 dataset fixes inconsistencies in energy and force labels.
Each configuration shares the same atomic number array (`nuclear_charges`), making it suitable for:

- Molecular potential energy surface (PES) modeling
- Machine learning force field (MLFF) training
- Graph neural network–based joint prediction of energy and forces

#### Dataset File Field Descriptions

| Field Name | Shape | Type | Description |
|-------------|--------|------|-------------|
| `nuclear_charges` | `(24,)` | int | Atomic numbers (Z) of each atom; constant across configurations |
| `coords` | `(N, 24, 3)` | float | Atomic coordinates for all configurations (Å) |
| `energies` | `(N,)` | float | Scalar potential energy for each configuration (kcal/mol) |
| `forces` | `(N, 24, 3)` | float | Atomic forces per configuration (kcal/mol/Å) |

### Core Code Implementation

- The main code modules are located in the `models` and `src` folders:

```text
applications
  └── nequip
        ├── rmd.yaml               # Configuration file
        ├── train.py               # Training entry script
        ├── predict.py             # Inference entry script
        ├── README.md              # English documentation
        ├── README_CN.md           # Chinese documentation
        ├── nequip.ipynb           # Chinese Notebook example
        ├── nequip_en.ipynb        # English Notebook example
        ├── images
        |     ├── architecture.png # Model architecture diagram
        |     └── result.png       # Result visualization
        ├── models
        |     ├── nequip.py        # Main model
        |     ├── convolution.py   # Equivariant convolution layer
        |     ├── embedding.py     # Atom embedding module
        |     ├── graph.py         # Graph structure construction
        |     ├── message_passing.py # Message passing layer
        |     └── utils.py         # Model utility functions
        └── src
              ├── dataset.py       # Dataset loading and preprocessing
              ├── trainer.py       # Training process encapsulation
              ├── predicter.py     # Inference and evaluation tools
              └── utils.py         # Logging and general utilities
```

## Model Running Steps

### Training

Before running, modify the training-related configurations in `rmd.yaml` (such as data path, batch size, learning rate, etc.) as needed. Then execute:

```bash
python train.py --config_file_path ./rmd.yaml --mode GRAPH --device_target Ascend --device_id 0 --dtype float32
```

Parameter description:

- `--config_file_path`: Path to the configuration file.
- `--mode`: Execution mode, `GRAPH` for high-performance static graph.
- `--device_target`: Target device type; `Ascend` is recommended for training.
- `--device_id`: Device index.
- `--dtype`: Calculation precision type (e.g., `float32`).

### Inference

After training and saving the model parameters, use `predict.py` to predict energy and forces:

```bash
python predict.py --config_file_path ./rmd.yaml --mode GRAPH --device_target GPU --device_id 0 --dtype float32
```

The parameters in the inference stage are basically the same as the training script. For more configurations, please refer to `rmd.yaml` and `predict.py`.

### Notebook Execution

You can execute the provided Jupyter Notebooks (both [English](nequip_en.ipynb) and [Chinese](nequip.ipynb) versions) step-by-step to train and validate the model interactively.

### Evaluation

The following figure shows the molecular energy prediction results. From the figure, it can be seen that the predicted results are basically consistent with the true results:

![alt text](images/result.png)

Example of training process logs:

```bash
python train.py --config_file_path ./rmd.yaml --mode GRAPH --device_target Ascend --device_id 0 --dtype float32
```

```log
2024-03-25 21:49:49 (INFO): ---- Configuration Summary -----
.
.
.
2024-03-25 21:49:49 (INFO): --------------------------------
2024-03-25 21:49:49 (INFO): Loading data...
2024-03-25 21:50:13 (INFO): Initializing model...
2024-03-25 21:50:37 (INFO): Initializing train...
2024-03-25 22:01:58 (INFO): epoch 1:  train loss: 1000.02729235, time gap: 680.55, total time used: 680.55
.
.
.
```

Example of notebook execution logs:

```log
Epoch 1/2: 100%
 190/190 [06:24<00:00,  1.01it/s]
.......
Epoch 2/2: 100%
 190/190 [03:33<00:00,  1.15s/it]
```

## License

- Open Source License: `Apache License 2.0`
- License Link: http://www.apache.org/licenses/LICENSE-2.0

## Citation

- If this project helps your research, please cite the related work:
    - Batzner S, Musaelian A, Sun L, et al. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials[J]. Nature Communications, 2022, 13: 2453.
