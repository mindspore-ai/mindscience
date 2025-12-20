ENGLISH | [CHINESE](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_mechanism_fusion/p2c2net/README_CN.md)

# Solving 2d Burgers Equation by Using P2C2Net

## Background Introduction

### Overview

**P2C2Net (PDE-Preserved Coarse Correction Network)** is a novel neural network architecture designed to efficiently solve spatiotemporal partial differential equations (PDEs) on coarse mesh grids with limited training data. Original paper is [P2C2Net: PDE-Preserved Coarse Correction Network for Efficient Prediction of Spatiotemporal Dynamics](https://arxiv.org/pdf/2411.00040).

![model architecture](https://raw.atomgit.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_mechanism_fusion/p2c2net/images/model_architecture.png)

As shown in the figure above, the model consists of two synergistic modules: (1) a trainable PDE block that learns to update the coarse solution (i.e., the system state), based on a high-order numerical scheme with boundary condition encoding, and (2) a neural network block that consistently corrects the solution on the fly. In particular, the model adopts a learnable symmetric Conv filter, with weights shared over the entire model, to accurately estimate the spatial derivatives of PDE based on the neural-corrected system state.

### Burger's Equation

The Burgers’ equation is a nonlinear PDE that models the propagation and reflection of shock waves. It is widely used in fluid mechanics, nonlinear acoustics, gas dynamics, and other fields.

$$
\frac{\partial \mathbf{u}}{\partial t}=\nu \nabla^2 \mathbf{u}-\mathbf{u}\cdot \nabla \mathbf{u}, t\in [0,T], x\in [0,1]^2
$$

Periodic boundaries are used to avoid non-physical reflections/errors caused by artificially specified computational domains, and are suitable for unbounded domain problems as well as periodic physical structures/phenomena. The core requirement is that physical quantities satisfy numerical equality and continuous derivatives on the "corresponding boundaries" of the computational domain.

$$
\mathbf{u}(\mathbf{x}_1, t)=\mathbf{u}(\mathbf{x}_2, t), \nabla\mathbf{u}(\mathbf{x}_1, t)=\nabla\mathbf{u}(\mathbf{x}_2, t)
$$

Where $\mathbf{x}_1\in \partial \Omega_1, \mathbf{x}_2\in \partial \Omega_2$ are the periodic corresponding points on the boundaries.

### Problem Descriptions

In this project, we focus on solving the **2D Burgers’ equation** efficiently using P2C2Net.

$$
\mathbf{u}_t \mapsto \mathbf{u}(\cdot, t+1)
$$

## Model Implementation

### Hardware Requirements

NPU memory>32G

### MindSpore & MindScience Version

mindspore>=2.5.0
mindscience==0.8.0

### Installation

1. Ensure that the correct versions of MindSpore and MindScience are installed in the environment;
2. Additional python packages, such as numpy、pandas、sympy、matplotlib, should be installed advancedly.
3. Clone the MindScience repository or obtain the codes directly from [MindFlow/applications/data_mechanism_fusion/p2c2net](https://atomgit.com/mindspore-lab/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/p2c2net);

### Dataset

Download train and test dataset: [MindFlow/applications/data_mechanism_fusion/p2c2net/src/data_gen.py](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_mechanism_fusion/p2c2net/src/data_gen.py).

### Coding

The specific process for solving this problem using MindFlow is as follows:

1. Create the dataset.
2. Train the model.
3. Check results.

#### 1. Create the dataset

The data generation codes can be downloaded from [dataGen.py](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_mechanism_fusion/p2c2net/src/data_gen.py).

First, generate training and testing data by running:

```shell
cd src
python dataGen.py
```

#### 2. Train the model

The training codes can be downloaded from [train_burgers.py](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_mechanism_fusion/p2c2net/train_burgers.py).

Run the following command to train P2C2Net on the generated data:

```shell
python p2c2net/train_burgers.py --experiment p2c2net
```

where

`--experiment` is the the experiment directory. It should include experiment specifications under 'config/';

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. Default 'GRAPH';

`--device_target` represents the type of computing platform used, which can be selected as 'Ascend' or 'GPU', default 'Ascend';

`--device_id` represents the calculation card number used, which can be filled in according to the actual situation, default 0;

`--continue` represents whether to resume training from a saved checkpoint, default False;

`--config_filename` is the name of the configuration file (under the `configs/` directory) that defines experiment settings such as model parameters, training schedule, default 'burgers.json';

`--train_stage` specifies whether to enable the training mode, default True;

`--test_stage` specifies whether to enable the testing mode, default True;

#### 3. Check results

After training, experiment outputs (checkpoints and evaluation results) are saved in result directory under the --experiment directory you provided. Use the saved checkpoints to reproduce evaluations or continue training.

## Experiment Results Display

![inference result](https://raw.atomgit.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_mechanism_fusion/p2c2net/images/inference.png)

## License

- License: `Apache License 2.0`
- License Link: `https://atomgit.com/mindspore-lab/mindscience/blob/master/LICENSE`

## Acknowledgments

### Contributors

gitee id: [liuguangyuu](https://gitee.com/liuguangyuu)

email: liuguangyuu@outlook.com

## Contact Us

If you have any suggestions for MindScience, please contact us via [issue](https://atomgit.com/mindspore-lab/mindscience/issues), and we will address them promptly.

## Quotation

If this project is helpful to your research, please cite the relevant work.

- Wang Q, Ren P, Zhou H, et al. P²C²Net: PDE-preserved coarse correction network for efficient prediction of spatiotemporal dynamics[C]//The Thirty-eighth Annual Conference on Neural Information Processing Systems. 2024.