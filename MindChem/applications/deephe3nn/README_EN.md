# Model Name

> DeepHE3nn

## Introduction

DeepHE3nn is an E3-equivariant neural network that uses atomic structures in crystals to predict the electronic Hamiltonian of the system.

## Dataset

Download `Bilayer_graphene_dataset.zip` from https://zenodo.org/records/7553640 to the current directory and extract it. Do not change the file name.

## Environment Requirements

1. Install `mindspore>=2.7.1`
2. Install `mindscience`
3. Install dependencies: `pip install -r requirements.txt`

## Quick Start

1. Download the dataset to the current directory.
2. Training command: `python train.py configs/Bilayer_graphene_train.ini`

## Script Description

1. `train.py`: includes graph data generation and model training
2. `predict.py`: inference script

## Code Structure

```txt
deephe3nn
    │  README.md                       README (Chinese)
    │  README_EN.md                    README (English)
    │  train.py                        Training entry
    │  predict.py                      Inference entry
    │  requirements.txt                Python dependencies
    │  
    └─data
    │      data.py                     Dataset processing
    │      graph.py                    Graph data structures
    │  
    └─models
    │      kernel.py                   Main execution flow
    │      parse_configs.py            Config processing
    │      e3modules.py                E3 modules
    │      model.py                    Model definition
    │      utils.py                    Utility functions
    └─graph
    │      graph.py                    Graph operations
    │      loss.py                     Graph loss functions
    │  
    └─configs
           Bilayer_graphene_train.ini  Model config file
```

## Training and Inference

### Training

```bash
pip install -r requirements.txt
python train.py configs/Bilayer_graphene_train.ini
```

### Inference

Set the path to the checkpoint in the config under `checkpoint_dir`.

```bash
pip install -r requirements.txt
python predict.py configs/Bilayer_graphene_train.ini
```

### Training/Inference Logs

```log
INFO:root:Starting new training process
INFO:root:-------Begin training-------
INFO:root:=================================epoch: 0
.
.
.
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