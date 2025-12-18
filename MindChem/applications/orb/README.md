# Orb

## Background

In materials science, designing novel functional materials has always been a key part of emerging technologies. However, traditional ab initio calculation methods are slow in designing new inorganic materials and difficult to scale to systems of practical size. In recent years, deep learning methods have demonstrated their powerful capabilities in multiple fields, capable of running efficiently through parallel architectures. The core innovation of the ORB model lies in applying this deep learning approach to materials modeling, learning the complexity of interatomic interactions through a scalable graph neural network architecture. The ORB model is a machine learning force field (MLFF) based on graph neural networks (GNNs), designed as a universal interatomic potential model suitable for various simulation tasks (geometry optimization, Monte Carlo simulations, and molecular dynamics simulations). The input to the model is a graph structure containing atomic positions, types, and system configuration (such as unit cell size and boundary conditions); the outputs include the total energy of the system, force vectors for each atom, and unit cell stress. Compared to existing open-source neural network potential models (such as MACE), the ORB model achieves a 3-6 times speed improvement at large system scales. In the Matbench Discovery benchmark, the ORB model reduced errors by 31% compared to other methods and became the state-of-the-art model on this benchmark at the time of release. The ORB model performs excellently in zero-shot evaluation, remaining stable even in molecular dynamics simulations of high-temperature aperiodic molecules without fine-tuning for specific tasks.

![Orb model predicts free energy](docs/orb.png)

In the figure above: (a) Free energy surfaces of MACE + D3 (left) and Orb-D3 (right) obtained in Mg-MOF-74 using the Widom insertion method. The blue regions near open metal sites represent the lowest free energy, indicating these are the preferred adsorption sites for CO2. (b) Adsorption positions of CO2 in Mg-MOF-74, showing the two most favorable adsorption sites obtained via the Widom insertion method, with adsorption energies of -54.5 kJ/mol and -54.4 kJ/mol, respectively. Although the energy minimum positions predicted by Orb and MACE are similar, the free energy minimum of ORB is numerically closer to the experimentally measured adsorption heat (-44 kJ/mol).

## Model Implementation

### Hardware Requirements

- Runs on the MindSpore framework. The backend device can be selected via the `device_target` field in the config files (see `configs/config_eval.yaml` and `evaluate.py`), with `Ascend` as the default.

### Version Dependencies

- Requires `MindSpore == 2.7.0` (see `requirement.txt`).
- Requires `MindScience` to provide foundational components for equivariant computations.
- Other Python dependencies are listed in `requirement.txt`.

### Installation

- Install MindSpore: follow the official installation guide at `https://www.mindspore.cn/install`
- Install MindScience: see `https://atomgit.com/mindspore-lab/mindscience`
- Install Python dependencies: `pip install -r requirement.txt`

### Dataset

- Download the training and test datasets from the [dataset link](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/dataset/) and place them under the `dataset` folder in the current path (create it manually if it does not exist).
- Download the Orb pre-trained checkpoint `orb-mptraj-only-v2.ckpt` from the [model link](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/orb_ckpts/) and place it under the `orb_ckpts` folder in the current path (create it manually if it does not exist).

Example directory structure:

```text
orb                                                 # ORB fine-tuning project
├── dataset
│   ├── train_mptrj_ase.db                           # Training dataset for fine-tuning (ASE trajectories, SQLite)
│   └── val_mptrj_ase.db                             # Validation / test dataset for fine-tuning
│
├── orb_ckpts                                        # Directory for pre-trained & fine-tuned checkpoints
│   └── orb-mptraj-only-v2.ckpt                      # Pre-trained ORB checkpoint (mptraj-only task)
│
├── configs                                          # Config files for training / inference
│   ├── config.yaml                                  # Single-card training configuration (lr, batch_size, etc.)
│   ├── config_parallel.yaml                         # Multi-card data-parallel training configuration
│   └── config_eval.yaml                             # Inference / evaluation configuration
│
├── src                                              # Core code for data processing and training
│   ├── ase_dataset.py                               # Load and wrap ASE datasets (read SQLite, build atomic graphs)
│   ├── atomic_system.py                             # Data structures for atomic systems (positions, species, cell, etc.)
│   ├── base.py                                      # Common base classes and utilities (e.g., batch_graphs)
│   ├── featurization_utilities.py                   # Tools to convert atomic systems into model input features
│   ├── pretrained.py                                # Interfaces for building and loading pre-trained ORB models
│   ├── property_definitions.py                      # Config and naming rules for energy / forces / stress, etc.
│   ├── trainer.py                                   # Training loop and loss wrappers (e.g., OrbLoss)
│   ├── segment_ops.py                               # Segment-wise reduction ops (segment_sum / mean / max)
│   └── utils.py                                     # Utility functions (seeding, logging, optimizer & LR scheduler)
│
├── models                                           # Model definitions (GNN / ORB networks)
│    ├── __init__.py                                 # Package initializer for orb
│    ├── gns.py                                      # GNS (Graph Network Simulator) related structures / APIs
│    ├── orb.py                                      # Main ORB architecture (encoder + heads)
│    └── utils.py                                    # Internal utilities and helper modules for ORB
│
├── finetune.py                                      # Entry script for model fine-tuning
├── evaluate.py                                      # Entry script for model inference / evaluation
│
├── run.sh                                           # Single-card training launcher (wraps finetune.py + config.yaml)
├── run_parallel.sh                                  # Multi-card training launcher (msrun + config_parallel.yaml)
└── requirement.txt                                  # Python dependency list for environment setup
```

- The core model implementation is composed of modules under `src` and `models`: `OrbLoss` and the training loop are implemented in `finetune.py` and `src/trainer.py`, while `evaluate.py` provides the inference and evaluation entry point.

## Running the Model

### Training

#### Single-card Training

- Modify the training parameters in `configs/config.yaml`:
    - Set the training and validation datasets for the fine-tuning stage via `train_data_path` and `val_data_path`.
    - Set the checkpoint directory to load the pre-trained model via `checkpoint_path`.
- After configuration, run:

```bash
pip install -r requirement.txt
bash run.sh
```

The training logs will look similar to:

```log
==============================================================================================================
Please run the script as:
bash run.sh
==============================================================================================================
Loading datasets: dataset/train_mptrj_ase.dbTotal train dataset size: 800 samples
Loading datasets: dataset/val_mptrj_ase.dbTotal train dataset size: 200 samples
Model has 25213610 trainable parameters.
Epoch: 0/100,
 train_metrics: {'data_time': 0.00010895108183224995, 'train_time': 386.58018293464556, 'energy_reference_mae': 5.598883946736653, 'energy_mae': 3.3611322244008384, 'energy_mae_raw': 103.14391835530598, 'stress_mae': 41.36046473185221, 'stress_mae_raw': 12.710869789123535, 'node_mae': 0.02808943825463454, 'node_mae_raw': 0.0228044210622708, 'node_cosine_sim': 0.7026202281316122, 'fwt_0.03': 0.23958333333333334, 'loss': 44.74968592325846}
 val_metrics: {'energy_reference_mae': 5.316623687744141, 'energy_mae': 3.594848871231079, 'energy_mae_raw': 101.00129699707031, 'stress_mae': 30.630516052246094, 'stress_mae_raw': 9.707925796508789, 'node_mae': 0.017718862742185593, 'node_mae_raw': 0.014386476017534733, 'node_cosine_sim': 0.5506304502487183, 'fwt_0.03': 0.375, 'loss': 34.24308395385742}

...

Epoch: 99/100,
 train_metrics: {'data_time': 7.802306208759546e-05, 'train_time': 59.67856075416785, 'energy_reference_mae': 5.5912095705668134, 'energy_mae': 0.007512244085470836, 'energy_mae_raw': 0.21813046435515085, 'stress_mae': 0.7020445863405863, 'stress_mae_raw': 2.222463607788086, 'node_mae': 0.04725319395462672, 'node_mae_raw': 0.042800972859064736, 'node_cosine_sim': 0.3720853428045909, 'fwt_0.03': 0.09895833333333333, 'loss': 0.7568100094795227}
 val_metrics: {'energy_reference_mae': 5.308632850646973, 'energy_mae': 0.27756747603416443, 'energy_mae_raw': 3.251189708709717, 'stress_mae': 2.8720269203186035, 'stress_mae_raw': 9.094478607177734, 'node_mae': 0.05565642938017845, 'node_mae_raw': 0.05041291564702988, 'node_cosine_sim': 0.212838813662529, 'fwt_0.03': 0.19499999284744263, 'loss': 3.2052507400512695}
Checkpoint saved to orb_ckpts/
Training time: 7333.08717 seconds
```

#### Multi-card Parallel Training

- Modify the training parameters in `configs/config_parallel.yaml` and `run_parallel.sh`:
    - Set the training and validation datasets for the fine-tuning stage via `train_data_path` and `val_data_path`.
    - Set the checkpoint directory to load the pre-trained model via `checkpoint_path`.
    - For other training settings, refer to the Training Configuration section.
    - Modify `--worker_num` and `--local_worker_num` in `run_parallel.sh` to set the number of devices to use.

```bash
pip install -r requirement.txt
bash run_parallel.sh
```

The training logs will look similar to:

```log
Loading datasets: dataset/train_mptrj_ase.dbTotal train dataset size: 800 samples
Loading datasets: dataset/train_mptrj_ase.dbTotal train dataset size: 800 samples
Loading datasets: dataset/train_mptrj_ase.dbTotal train dataset size: 800 samples
Loading datasets: dataset/train_mptrj_ase.dbTotal train dataset size: 800 samples
Loading datasets: dataset/val_mptrj_ase.dbTotal train dataset size: 200 samples
Loading datasets: dataset/val_mptrj_ase.dbTotal train dataset size: 200 samples
Loading datasets: dataset/val_mptrj_ase.dbTotal train dataset size: 200 samples
Loading datasets: dataset/val_mptrj_ase.dbTotal train dataset size: 200 samples
Model has 25213607 trainable parameters.
Model has 25213607 trainable parameters.
Model has 25213607 trainable parameters.
Model has 25213607 trainable parameters.

...

Training time: 2375.89474 seconds
Training time: 2377.02413 seconds
Training time: 2377.22778 seconds
Training time: 2376.63176 seconds
```

Under the same training configuration, multi-card parallel training achieves significant performance improvement compared to single-card training (based on the example logs above):

- Single-card training time: about 7,300 s
- 4-card parallel training time: about 2,400 s
- Performance improvement: about 67%
- Speedup ratio: about 3×

### Inference / Evaluation

- Modify the inference / evaluation parameters in `configs/config_eval.yaml`:
    - Set the test dataset via `val_data_path`.
    - Set the pre-trained or fine-tuned checkpoint to load via `checkpoint_path`.
    - For other evaluation settings, refer to the Evaluating Configuration section.

Run:

```bash
python evaluate.py
```

The logs will look similar to:

```log
Loading datasets: dataset/val_mptrj_ase.dbTotal train dataset size: 200 samples
Model has 25213607 trainable parameters.
.Validation loss: 0.89507836
    energy_reference_mae: 5.3159098625183105
    energy_mae: 0.541229784488678
    energy_mae_raw: 4.244375228881836
    stress_mae: 0.22862032055854797
    stress_mae_raw: 10.575761795043945
    node_mae: 0.12522821128368378
    node_mae_raw: 0.04024107754230499
    node_cosine_sim: 0.38037967681884766
    fwt_0.03: 0.22499999403953552
    loss: 0.8950783610343933
```
