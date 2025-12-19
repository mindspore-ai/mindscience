# MatFormer

## Background

[MatFormer](https://arxiv.org/abs/2209.11807) is a state-of-the-art model
based on **Graph Neural Networks (GNNs)** and the **Transformer**
architecture, designed for predicting properties of crystalline materials.

By operating on periodic crystal graphs, MatFormer captures both local and
global structural information while remaining robust to lattice translations
and other symmetries. Compared with classical models such as CGCNN, SchNet,
and MEGNet, MatFormer achieves superior accuracy on tasks including formation
energy per atom, bandgap, and lattice-related properties.

## Model Implementation

### Hardware Requirements

- Scripts are currently configured to run on `Ascend` devices. The target
  device and `device_id` are specified in `config.yaml` under the `train`
  section and used in `train.py`.

### Version Requirements

- `MindSpore >= 2.7.0`
- `MindScience` (for scientific and equivariant computation components)

### Installation

- Install MindSpore: see the official guide at
  `https://www.mindspore.cn/install`
- Install MindScience: see `https://atomgit.com/mindspore-lab/mindscience`
- Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Dataset

- Download `jdft_3d-12-12-2022.json` from
  <https://figshare.com/articles/dataset/jdft_3d-7-7-2018_json/6815699> to
  the current directory without changing the file name.

#### Basic Information

- **Dataset name**: `jdft_3d-12-12-2022.json`
- **Source**: [JARVIS-DFT](https://jarvis.nist.gov/)
  (Joint Automated Repository for Various Integrated Simulations – Density Functional Theory)
- **Size**: **75,993** 3D bulk crystal structures
- **Format**: JSON
- **Material identifier**: unique `jid` (e.g., `JVASP-90856`)

#### Dataset Overview

The dataset contains **3D bulk crystalline materials** with properties
computed via **Density Functional Theory (DFT)**. It is suitable for
materials discovery, property prediction, and machine-learning benchmarks.

#### Key Fields

| Field | Type | Description |
|------|------|-------------|
| `jid` | str | Unique JARVIS material ID (e.g., `JVASP-90856`) |
| `formula` | str | Chemical formula (e.g., `TiCuSiAs`) |
| `spg_number` / `spg_symbol` | int / str | Space group number and symbol (e.g., 129, `P4/nmm`) |
| `formation_energy_peratom` | float | Formation energy per atom (eV/atom); more negative means more stable |
| `optb88vdw_bandgap` | float | Bandgap computed with the OptB88vdW functional (eV) |
| `mbj_bandgap`, `hse_gap` | float | Bandgaps from mBJ or HSE06 functionals (available for some materials) |
| `atoms` | dict | **Core structural data**: <br>• `lattice_mat`: 3×3 lattice matrix<br>• `coords`: atomic coordinates<br>• `elements`: list of elements<br>• `cartesian`: whether coordinates are Cartesian (bool) |
| `density` | float | Material density (g/cm³) |
| `ehull` | float | Energy above convex hull (eV/atom); < 0.1 eV/atom is typically considered stable |
| `func` | str | DFT functional used (e.g., `OptB88vdW`) |
| `dimensionality` | str | Material dimensionality (all entries are `3D-bulk`) |
| `crys` | str | Crystal system (e.g., `tetragonal`, `cubic`) |
| `nat` | int | Total number of atoms |
| `reference` | str | Corresponding Materials Project ID (e.g., `mp-1080455`) |

### Core Code

- The main modules are under the `data` and `models` folders:

```text
applications
  └── matformer
        ├── README.md                    # README (English)
        ├── README_CN.md                 # README (Chinese)
        ├── config.yaml                  # Configuration file
        ├── train.py                     # Training entry
        ├── predict.py                   # Inference entry
        ├── requirements.txt             # Python dependencies
        ├── matformer_application.ipynb  # Jupyter notebook (Chinese)
        ├── matformer_application_EN.ipynb  # Jupyter notebook (English)
        ├── data
        |     ├── __init__.py            # Package init
        |     ├── data.py                # JARVIS data loading and graph construction
        |     ├── features.py            # Feature engineering utilities
        |     ├── generate.py            # Dataset preprocessing and splitting
        |     └── graphs.py              # Crystal graph and dataset definitions
        ├── models
        |     ├── __init__.py
        |     ├── matformer.py           # Main MatFormer network
        |     ├── transformer.py         # MatFormerConv and Transformer blocks
        |     ├── utils.py               # RBF expansion, LR scheduler, loss recorder
        |     └── graph
        |           ├── __init__.py
        |           ├── dataloader.py    # Graph DataLoader
        |           ├── graph.py         # Graph operations and global aggregation
        |           ├── loss.py          # Masked L1/L2 loss
        |           └── normlization.py  # Normalization utilities
        └── images
              ├── architecture.png       # Model architecture diagram
              └── result.png             # Example prediction results
```

- The main model is implemented in `Matformer` (`models/matformer.py`): RBF
  expansion and stacked MatFormerConv layers update node and edge features on
  the periodic crystal graph, followed by graph-level pooling to obtain
  property predictions. The training pipeline is organized in `train.py`,
  including data preparation (`data/generate.py`), graph data loading
  (`models/graph/dataloader.py`), learning-rate scheduling (`OneCycleLr`),
  and loss tracking (`LossRecord`).

### Model Architecture

![architecture](images/architecture.png)

The overall workflow of MatFormer is:

1. **Input feature extraction**:
  - Atomic features $ a_i $ are embedded via a dense layer to obtain
     initial node representations $ f^*_i $.
  - Interatomic distances $ d_{ij}^h $ are expanded into Gaussian radial
     basis features $ e_{ij}^h $ using an RBF kernel.
  - Edge features $ e_{ij}^h $ are further processed by a Linear + Softplus
     block and used as edge inputs.

2. **Stacked MatFormerConv layers**:
  - Multiple MatFormerConv layers are stacked; each layer performs attention
     on the periodic crystal graph to update node and edge representations.

3. **Readout**:
  - Final node representations are aggregated via mean pooling over nodes.
  - A Linear → SiLU → Linear head produces the final property prediction
     (e.g., formation energy per atom).

## Running the Model

### Training

- Ensure the following preparations are completed:
- MindSpore, MindScience, and Python dependencies are installed.
- `jdft_3d-12-12-2022.json` is downloaded into the current directory.
- Training parameters are configured in `config.yaml`, including:
- `train.device`, `train.device_id`: target device settings.
- `train.props`: target property (e.g. `formation_energy_peratom`).
- `train.epoch_size`, `train.batch_size`: number of epochs and batch size.
- `train.dataset_dir`, `train.ckpt_dir`: paths for preprocessed data and checkpoints.

Run training from the `matformer` directory:

```bash
pip install -r requirements.txt
python train.py
```

During training, raw JARVIS data are converted into graph representations and
cached under `dataset_dir`, and model checkpoints are saved in `ckpt_dir` as
configured in `config.yaml`.

### Inference

- Set the path of the checkpoint to load in the `predictor.checkpoint_path`
  field of `config.yaml` (default `./ckpt/best_matformer.ckpt`).
- Run prediction from the `matformer` directory:

```bash
python predict.py
```

Inference-related options (such as number of epochs) are controlled by the
`predictor` section in `config.yaml`. Prediction results are printed in the
logs and can be saved or post-processed as needed.

### Notebook Workflow

You can also run training and evaluation step-by-step using the provided
Jupyter notebooks:

- `matformer_application.ipynb`: Chinese notebook
- `matformer_application_EN.ipynb`: English notebook

## Results

The figure below shows formation energy predictions from a fully trained
MatFormer model. Predicted values closely match the ground truth, indicating
low prediction error.

![result](images/result.png)

### Example Training Logs

Example log output from `train.py`:

```log
INFO:root:Loading from saved file...
INFO:root:The model you built has 2786689 parameters.
INFO:root:load from existing check point................
INFO:root:finish load from existing checkpoint, start training from epoch: 1
INFO:root:change learning rate to current step: 953
INFO:root:current learning rate: 9.345746e-08
INFO:root:Start to initialise train_loader
INFO:root:Start to initialise eval_loader
INFO:root:+++++++++++++++ start traning +++++++++++++++++++++
INFO:root:==============================step: 0 ,epoch: 0
INFO:root:learning rate: 9.345746e-08
INFO:root:train mse loss: 0.09808009
INFO:root:is_finite: True
INFO:root:traning time: 22.13266158103943
...
INFO:root:step:117, epoch: 499
INFO:root:validation mse loss: 0.004059551
INFO:root:validation mae loss: 0.034488887
INFO:root:validation time: 0.041112422943115234
INFO:root:epoch 499 running time: 137.772692
INFO:root:epoch 499 average train mse loss: 0.0003474082
INFO:root:epoch 499 average validation mse loss: 0.00414170
INFO:root:epoch 499 average validation mae loss: 0.03259226
```

Example log output from the Jupyter notebook:

```log
Model trainable parameters: %s 2786689
Starting new training process
.Saved best model at epoch %d, MSE: %.6f 0 0.0991247
Epoch 0 | Train MSE: 0.152263 | Val MSE: 0.099125 | Val MAE: 0.211994
```

## License

- License: `Apache License 2.0`
- License link: `http://www.apache.org/licenses/LICENSE-2.0`

## Citation

- If this project is helpful to your research, please cite, for example:
    - Yan K, et al. Periodic graph transformers for crystal material property
      prediction\[J\]. Advances in Neural Information Processing Systems, 2022.
