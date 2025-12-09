# Orb

## Overview

> In materials science, designing novel functional materials has always been a key part of emerging technologies. However, traditional ab initio calculation methods are slow in designing new inorganic materials and difficult to scale to systems of practical size. In recent years, deep learning methods have demonstrated their powerful capabilities in multiple fields, capable of running efficiently through parallel architectures. The core innovation of the ORB model lies in applying this deep learning approach to materials modeling, learning the complexity of interatomic interactions through a scalable graph neural network architecture. The ORB model is a machine learning force field (MLFF) based on graph neural networks (GNNs), designed as a universal interatomic potential model suitable for various simulation tasks (geometry optimization, Monte Carlo simulations, and molecular dynamics simulations). The input to the model is a graph structure containing atomic positions, types, and system configuration (such as unit cell size and boundary conditions); the outputs include the total energy of the system, force vectors for each atom, and unit cell stress. Compared to existing open-source neural network potential models (such as MACE), the ORB model achieves a 3-6 times speed improvement at large system scales. In the Matbench Discovery benchmark, the ORB model reduced errors by 31% compared to other methods and became the state-of-the-art model on this benchmark at the time of release. The ORB model performs excellently in zero-shot evaluation, remaining stable even in molecular dynamics simulations of high-temperature aperiodic molecules without fine-tuning for specific tasks.

![Orb model predicts free energy](docs/orb.png)

> In the figure above: (a) Free energy surfaces of MACE + D3 (left) and Orb-D3 (right) obtained in Mg-MOF-74 using the Widom insertion method. The blue regions near open metal sites represent the lowest free energy, indicating these are the preferred adsorption sites for CO2. (b) Adsorption positions of CO2 in Mg-MOF-74, showing the two most favorable adsorption sites obtained via the Widom insertion method, with adsorption energies of -54.5 kJ/mol and -54.4 kJ/mol, respectively. Although the energy minimum positions predicted by Orb and MACE are similar, the free energy minimum of ORB is numerically closer to the experimentally measured adsorption heat (-44 kJ/mol).

## Environment Requirements

> 1. Install `mindspore (2.7.0)`
> 2. Install dependencies: `pip install -r requirement.txt`

## Quick Start

> 1. Download the corresponding dataset from [dataset link](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/dataset/) and place it in the `dataset` directory
> 2. Download the orb pre-trained model ckpt from [model link](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/orb_ckpts/) and place it in the `orb_ckpts` directory
> 3. Install dependencies: `pip install -r requirement.txt`
> 4. Single-card training command: `bash run.sh`
> 5. Multi-card training command: `bash run_parallel.sh`
> 6. Evaluation command: `python evaluate.py`
> 7. Model prediction results will be stored in the `results` directory

### Code Directory Structure

```text
The main code modules are in the src folder, with the dataset folder containing the datasets, the orb_ckpts folder containing pre-trained models and trained model weight files, and the configs folder containing parameter configuration files for each code.

orb_models                                           # ORB pre-training / fine-tuning project
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
│   ├── __init__.py                                  # Package initializer for src
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
│    ├── __init__.py                          # Package initializer for orb
│    ├── gns.py                               # GNS (Graph Network Simulator) related structures / APIs
│    ├── orb.py                               # Main ORB architecture (encoder + heads)
│    └── utils.py                             # Internal utilities and helper modules for ORB
│
├── finetune.py                                      # Entry script for model fine-tuning
├── evaluate.py                                      # Entry script for model inference / evaluation
│
├── run.sh                                           # Single-card training launcher (wraps finetune.py + config.yaml)
├── run_parallel.sh                                  # Multi-card training launcher (msrun + config_parallel.yaml)
└── requirement.txt                                  # Python dependency list for environment setup

```  

## Download Dataset

Download the training and test datasets from [dataset link](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/dataset/) and place them in the dataset folder under the current path (create manually if it does not exist); download the orb pre-trained model `orb-mptraj-only-v2.ckpt` from [model link](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/orb/orb_ckpts/) and place it in the orb_ckpts folder under the current path (create manually if it does not exist); refer to [Code Directory Structure](#code-directory-structure) for file paths

## Training Process

### Single-Card Training

Modify the training parameters in the `configs/config.yaml` file:

> 1. Set the training and test datasets for the fine-tuning stage, see the `data_path` field
> 2. Set the pre-trained model weight file to load for training, modify the `checkpoint_path` path field
> 3. Other training settings see the Training Configuration section

```bash
pip install -r requirement.txt
bash run.sh
```

The code running results are as follows:

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

### Multi-Card Parallel Training

Modify the training parameters in the `configs/config_parallel.yaml` and `run_parallel.sh` files:

> 1. Set the training and test datasets for the fine-tuning stage, see the `data_path` field
> 2. Set the pre-trained model weight file to load for training, modify the `checkpoint_path` path field
> 3. Other training settings see the Training Configuration section
> 4. Modify `--worker_num=4 --local_worker_num=4` in the `run_parallel.sh` file to set the number of cards to use

```bash
pip install -r requirement.txt
bash run_parallel.sh
```

The code running results are as follows:

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

Under the same training configuration, parallel training achieved significant performance improvement compared to single-card training:

- Single-card training time: 7293.28995 seconds
- 4-card parallel training time: 2377.22778 seconds
- Performance improvement: 67.40%
- Speedup ratio: 3.07x

### Inference

Modify the inference parameters in the `configs/config_eval.yaml` file:

> 1. Set the test dataset, see the `val_data_path` field
> 2. Set the pre-trained model weight file to load for inference, modify the `checkpoint_path` path field
> 3. Other training settings see the Evaluating Configuration section

```bash
python evaluate.py
```

The code running results are as follows:

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