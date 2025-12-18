# DeephE3nn

## Background

DeephE3nn is an E(3)-equivariant neural network for accurately predicting the
electronic Hamiltonian of a system from the atomic configuration in crystals.

In traditional first-principles calculations, each atomic structure requires a
new solution of the Hamiltonian using high-cost methods such as density
functional theory. The computational cost grows rapidly with system size,
making it difficult to support large-scale materials screening and
high-throughput simulations.

By explicitly modeling rotational and translational symmetries in space via an
equivariant graph neural network, DeephE3nn efficiently learns the mapping
from “crystal structure → electronic Hamiltonian”. It significantly reduces
the computational cost while preserving physical symmetries. This example is
based on a bilayer graphene dataset and predicts the electronic Hamiltonian of
the material system, providing efficient approximations for downstream tasks
such as band structure and transport-property calculations.

## Model Implementation

### Hardware Requirements

- The scripts are configured to run on `Ascend` devices by default. Use the
  command-line argument `-device_id` to specify the device ID (see `train.py`
  and `predict.py`).

### Version Requirements

- `MindSpore >= 2.7.0`
- `MindScience >= 0.8.0` (for `mindscience.e3nn` and related modules)

### Installation

- Install MindSpore: see the official guide at
  `https://www.mindspore.cn/install`
- Install MindScience: see `https://atomgit.com/mindspore-lab/mindscience`
- Install Python dependencies:
  `pip install -r requirements.txt`

### Dataset

- Download `Bilayer_graphene_dataset.zip` from the
  [Zenodo page](https://zenodo.org/records/7553640) to the current directory
  and unzip it without changing the file name.

Example directory structure after extraction (illustrative):

```txt
deephe3nn
    ├─Bilayer_graphene_dataset
    │      ...
    └─configs
           Bilayer_graphene_train.ini
```

### Core Code

- The main modules are under the `data`, `graph`, and `models`      folders, and depend on MindScience modules such as `mindscience.e3nn`:

```text
applications
  └── deephe3nn
        ├── README.md                     # README (Chinese)
        ├── README_EN.md                  # README (English)
        ├── train.py                      # Training entry
        ├── predict.py                    # Inference entry
        ├── requirements.txt              # Environment dependencies (3rd‑party Python)
        ├── configs
        │     └── Bilayer_graphene_train.ini   # Training/inference configuration
        ├── data
        │     ├── __init__.py             # Package init
        │     ├── data.py                 # Dataset loading and preprocessing
        │     └── graph.py                # Graph data structures (data‑side)
        ├── graph
        │     ├── graph.py                # Graph structures and operators (model‑side)
        │     └── loss.py                 # Graph‑related loss functions
        └── models
              ├── default_configs         # Default config templates
              │     ├── base_default.ini
              │     ├── eval_default.ini
              │     └── train_default.ini
              ├── __init__.py
              ├── e3modules.py            # E(3)‑equivariant modules
              ├── kernel.py               # DeepHE3Kernel: training/evaluation pipeline
              ├── model.py                # Main network Net
              ├── parse_configs.py        # Config parsing utilities
              └── utils.py                # Training utilities (basis functions, etc.)
```

- The main model is composed of `Net` in `models/model.py` and the
  equivariant modules in `models/e3modules.py`. The training pipeline is
  wrapped by `DeepHE3Kernel` in `models/kernel.py`, including data loading,
  loss computation (e.g. `L2LossMask`), learning-rate scheduling, and logging.

## Running the Model

### Training

- Make sure the following preparations are completed:
    - MindSpore, MindScience, and all dependencies are installed.
    - `Bilayer_graphene_dataset.zip` is downloaded and extracted into the
      current directory.
    - Training parameters (such as batch size, learning rate, number of epochs,
      `checkpoint_dir`, etc.) are configured in
      `configs/Bilayer_graphene_train.ini`.
- Then run the following in the `deephe3nn` directory:

```bash
python train.py configs/Bilayer_graphene_train.ini
```

During training, model checkpoints are saved under the `checkpoint_dir`
specified in the config file, and training/validation losses are printed in
the log.

### Inference

- Set the path of the checkpoint to load in the `checkpoint_dir` field of the
  config file.
- Run the following in the `deephe3nn` directory:

```bash
python predict.py configs/Bilayer_graphene_train.ini
```

The inference script computes the electronic Hamiltonian for the given
structures and prints evaluation results in the logs. The exact output format
can be adjusted through the configuration and downstream task requirements.

### Example Training Log

```log
INFO:root:Starting new training process
INFO:root:-------Begin training-------
INFO:root:=================================epoch: 0
...
INFO:root:----------------------eval epoch: 916-------step: 19
INFO:root:evaluating time: 0.25410914421081543
INFO:root:learning rate: 3.159372e-10
INFO:root:val mse loss: 7.4168706e-06
INFO:root:epoch: 916

INFO:root:last train loss: 7.4168706e-06
INFO:root:average eval loss: 6.1306587e-06
INFO:root:Train finished, cost 63180.765609025955 s
INFO:root:best loss: 6.1306587e-06
```

## License

- Open-source license: `Apache License 2.0`
- License link: `http://www.apache.org/licenses/LICENSE-2.0`

## Citation

- If this project is helpful to your research, please cite the related work:
    - Xiaoxun Gong, He Li, Nianlong Zou, et al. General framework for E(3)-equivariant neural network representation of density functional theory Hamiltonian[J]. Nature Communications, 2023, 14: 2848.
