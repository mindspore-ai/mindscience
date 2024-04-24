# PDEformer-1: A Fundamental Model for One-dimensional PDEs

[![demo_video](images/video.png)](https://www.bilibili.com/video/BV1wm411C7Sq)

## Overview

### Problem Description

The dynamics of Partial Differential Equations (PDEs) are associated with a wide range of physical phenomena and engineering applications, such as wing design, electromagnetic field simulations, and stress analysis. These real-world engineering applications all require multiple invocations of PDE solvers. Although traditional methods of solving PDEs usually exhibit high accuracy, they often require substantial computational resources and time. However, to design a universal solver for all types of PDEs could be challenging.
In recent years, the Neural Operator approach, which employs neural networks to learn data from a large number of PDE solutions to approximate the PDE operators, has significantly improved the speed of solving the forward problems of PDEs. At the same time, the trained neural network model can also serve as a differentiable surrogate model to address inverse problems.
However, current neural operator methods still struggle to generalize to new types of PDEs, and training for new PDEs often faces high training costs and difficulties in data acquisition.

To address these issues, we introduce the PDEformer model, a neural operator model that can directly input any form of PDEs.
After being trained on a large-scale dataset of one-dimensional PDEs, the PDEformer model can quickly and accurately solve any form of one-dimensional PDE forward problems, and its zero-shot prediction accuracy within the training data distribution is higher than that of any expert model trained specifically for one type of equation (such as FNO, DeepONet).
The PDEformer requires no retraining for new problems within the equation coefficient distribution and can rapidly generalize to downstream tasks through few-shot learning with minimal data for cases outside the coefficient distribution.
At the same time, the PDEformer can be directly applied to solving inverse problems.
With further increases in dataset scale, the PDEformer is expected to become a foundational model for solving various PDE problems, propelling the development of scientific research and engineering applications.

## Technical Path

We consider one-dimensional time-dependent partial differential equations (PDEs) defined on $(t,x) \in [0,1] \times [-1,1]$, whose general form can be written as:

$$ \mathcal{F}(u_1,u_2,\dots,c_1,c_2,\dots,s_1(x),s_2(x),\dots)=0,$$

where $c_1,c_2,\dots \in \mathbb{R}$ are real-valued coefficients, $s_1(x),s_2(x)\dots$ are scalar functions (which can act as initial conditions, coefficient fields, etc.), and $u_1,u_2,\dots$ are the various components of the physical fields to be solved.
Here, we assume that the operator $\mathcal{F}$ has a symbolic expression, which may involve differential and algebraic operations.
The goal of PDEformer is to construct a surrogate model of the equation solution, in the form $(\mathcal{F},c_1,c_2,\dots,s_1(x),s_2(x),\dots)\mapsto (u_1,u_2,\dots)$,
which takes the symbolic form $\mathcal{F}$ of the PDE and the numerical information involved $c_1,c_2,\dots,s_1(x),s_2(x),\dots$ as inputs, and outputs the predicted solution $u_1,u_2,\dots$ for the corresponding equation.
Let's take the convection equation (single component) with periodic boundary conditions $u_t+cu_x=0$, $u(0,x)=g(x)$ as an example:

![](images/PDEformerV2Arch.png)

### Constructing the PDE Computational Graph

First, we represent $\mathcal{F}$ (the symbolic information about the PDE form to be solved) as a computational graph.
In this computational graph, a node can represent an unknown field component (denoted as `UF`), a scalar coefficient (`SC`), a coefficient field (`CF`), an initial condition (`IC`), as well as differential or algebraic operations, while a directed edge is used to specify the operands involved in an operation.
This forms a Directed Acyclic Graph (DAG) with heterogeneous nodes and homogeneous edges.

Next, to reflect the numerical information involved in the PDE, we assign a $d_e$-dimensional input feature to each node in the graph.
For a scalar coefficient $c$, its numerical value is input into a scalar encoder, and the $d_e$-dimensional output from the encoder serves as the input feature for the corresponding `SC` node.
Considering that scalar functions $s(x)$ contain relatively richer information, we introduce $N$ new "branch" nodes (of types $\mathtt{b}_1, \mathtt{b}_2, \dots, \mathtt{b}_N$) for each such scalar function, using the input features of these $N$ nodes to represent the numerical information contained in $s(x)$.
Specifically, we use a function encoder, whose input is a series of scattered points $\{(x_i,s(x_i))\}$ (the positions and distribution of points can be arbitrarily chosen), and the output $d_eN$-dimensional vector is used to provide input features for the $N$ branch nodes.
These branch nodes are connected with edges to the `IC` or `CF` vertex corresponding to $s(x)$.
The input features for all remaining nodes are set to zero (for simplification in programming implementation, actually the corresponding output when the scalar encoder receives a zero input).

Furthermore, we introduce $L$ additional nodes (of types $\mathtt{m}_1, \mathtt{m}_2, \dots, \mathtt{m}_L$) for each field component to be solved and connect them with the respective `UF` nodes.
When processing the graph data with a graph Transformer later, the information about the predicted solution for this field component will be aggregated into the embedding vectors corresponding to these new nodes.

For more implementation details and examples of the PDE computational graph, please refer to [docs/PDE_DAG.md](docs/PDE_DAG.md).

### Encoding Graph Data

The graph data obtained in the previous step contains symbolic and numerical information involved in the PDE.
We will integrate this information from the graph data and generate a latent encoding representing the solution for each field component $u_j$ to be solved

$$\pmb{\mu}_j = [{\mu}^1_j, \dots, {\mu}^L_j]^{\mathrm{T}} \in {\mathbb{R}}^{L \times d_e}.$$

This integration process uses a graph Transformer, which is a powerful type of graph neural network based on the Transformer architecture, adept at capturing and expressing complex graph structural information.
In the implementation, we have adopted the [Graphormer](https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html) architecture and made some suitable adjustments for the scenarios of PDE encoding.
In the output layer of the Graph Transformer, the embedding vector corresponding to nodes of type $\mathtt{m}_\ell$ (and associated with $u_j$) is denoted as $\mu_j^\ell \in \mathbb{R}^{d_e}$, which will participate in "modulating" the $\ell$-th hidden layer of the INR in the next step to generate the final predicted solution.

### Decoding to Obtain the Predicted Solution of the PDE

We use the Implicit Neural Representation (INR) approach to represent each component of the equation solution.
This network takes the spacetime coordinates $(t,x)$ as input and predicts the value of the corresponding equation solution component $\hat u_j(t,x)$ at those coordinates based on the latent code $\pmb{\mu}_j$.
The prediction provided in this manner does not depend on a specific discretization grid.
In the implementation, we choose the Poly-INR architecture and made the corresponding adaptations (see below figure).
The number of hidden layers in the Poly-INR is $L$, where the activation values of the $\ell$-th hidden layer will be "modulated" by $\mu_j^\ell$.
Note that all components in the PDE share the same set of INR parameters; they differ only in their corresponding latent encodings $\pmb{\mu}_j$.

![](images/HyperPolyINR2.png)

## Installation

First, ensure that MindSpore is successfully installed, as seen in the [Installation Tutorial](https://www.mindspore.cn/install).
Other dependencies can be installed with the following command:

```bash
pip3 install -r pip-requirements.txt
```

## File Directory

```text
./
│  inverse_function.py                           # Inverse Problem Code to reconstruct scalar functions in PDE (e.g., source term, velocity field in wave equation)
│  inverse_scalar.py                             # Inverse Problem Code to reconstruct scalar coefficients in PDE (e.g., diffusion coefficient in reaction-diffusion equation)
│  PDEformer_inference.ipynb                     # Interactive Model Inference Notebook (English)
│  PDEformer_inference_CN.ipynb                  # Interactive Model Inference Notebook (Chinese)
│  pip-requirements.txt                          # Python package dependencies
│  preprocess_data.py                            # Preprocessing Data and Saving to a New Auxiliary Data File
│  README.md                                     # README (English)
│  README_CN.md                                  # README (Chinese)
│  train.py                                      # Training Code
├─configs                                        # config directory
│  │  full_config_example.yaml                   # Full Configuration Example (Including all possible choices)
│  ├─baseline                                    # Baseline Model Config
│  │      advection-beta-0.1_fno.yaml            # Training FNO on Advection beta=0.1 dataset
│  │      burgers-nu-0.1_deeponet.yaml           # Training DeepONet on Burgers nu=0.1 dataset
│  │      reacdiff-nu-1.0-rho-1.0_unet.yaml      # Training U-Net on reaction-diffusion nu=1.0 rho=1.0 dataset
│  ├─finetune                                    # Finetune Model Config
│  │      burgers-nu-0.1_pdeformer-L.yaml        # Load pretrained L-scale PDEformer and finetune on Burgers nu=0.1 dataset
│  ├─inference                                   # Load pretrained PDEformer weights for inference
│  │      pdeformer-L.yaml                       # Inference Configuration for PDEformer with scale L
│  │      pdeformer-M.yaml                       # Inference Configuration for PDEformer with scale M
│  │      pdeformer-XL.yaml                      # Inference Configuration for PDEformer with scale XL
│  ├─inverse                                     # Pretrain Model on Inverse Problem Config
│  │      inverse_function.yaml                  # Config for Inverse Problem on Scalar Function (e.g., source term, velocity field in wave equation)
│  │      inverse_scalar.yaml                    # Config for Inverse Problem on Scalar Coefficients (e.g., diffusion coefficient in reaction-diffusion equation)
│  └─pretrain                                    # Pretrain Model Config
│         pdeformer-L.yaml                       # Config for L-scale PDEformer Training
│         pdeformer-M.yaml                       # Config for M-scale PDEformer Training
│         pdeformer-S.yaml                       # Config for S-scale PDEformer Training
│         pdeformer-XL.yaml                      # Config for XL-scale PDEformer Training
├─docs                                           # Additional Documentations
│      DATASET.md                                # Documentation for Dataset (English)
│      DATASET_CN.md                             # Documentation for Dataset (Chinese)
│      PDE_DAG.md                                # Documentation for PDE Computational Graph (English)
│      PDE_DAG_CN.md                             # Documentation for PDE Computational Graph (Chinese)
├─images                                         # Images for README
├─scripts                                        # Shell scripts for training, finetuning and inverse problem
│      run_distributed_train.sh                  # Shell script for distributed training
│      run_inverse_function.sh                   # Shell script for solving inverse problem on scalar function
│      run_inverse_scalar.sh                     # Shell script for solving inverse problem on scalar coefficients
│      run_standalone_train.sh                   # Shell script for single-GPU training
├─data_generation                                # Customized Data Generation Code
│      common.py                                 # Common functions for data generation
│      custom_multi_component.py                 # Multi-component PDE data generation main file
│      custom_sinus.py                           # Main file of PDE data generation with trigonometric function
│      custom_wave.py                            # Main file of PDE data generation with wave equation
│      inverse_sinus.py                          # Main file of inverse problem data generation with trigonometric function
│      inverse_wave.py                           # Main file of inverse problem data generation with wave equation
│      README.md                                 # README of data generation (English)
│      README_CN.md                              # README of data generation (Chinese)
└─src                                            # Basic Code Directory
    │  inference.py                              # Generate inference results for customized PDE
    ├─cell                                       # Code for the basic unit of the network
    │  │  basic_block.py                         # Basic block of the network
    │  ├─baseline                                # Baseline Model Code
    │  │      check_func.py                      # Check parameters
    │  │      deeponet.py                        # DeepONet network architecture
    │  │      dft.py                             # Discrete Fourier Transform
    │  │      fno2d.py                           # FNO-2d network architecture
    │  │      unet2d.py                          # UNet-2d network architecture
    │  └─pdeformer                               # PDEFormer Model Code
    │      │  function_encoder.py                # Encoder Block
    │      │  pdeformer.py                       # PDEFormer Network Architecture
    │      ├─graphormer                          # Graphormer Network Architecture
    │      │      graphormer_encoder.py          # Encoder Block
    │      │      graphormer_encoder_layer.py    # Encoder Layer
    │      │      graphormer_layer.py            # Encode Block of Node and Edge Information
    │      │      multihead_attention.py         # Multi-head Attention Block
    │      └─inr_with_hypernet                   # INR + HyperNet Model Code
    │              mfn.py                        # MFN + HyperNet Block
    │              siren.py                      # Siren + HyperNet Block
    │              poly_inr.py                   # Poly-INR + HyperNet Block
    ├─core                                       # Core Training Block
    │      losses.py                             # Loss Module
    │      lr_scheduler.py                       # lr_scheduler Module
    │      metric.py                             # Metric Module
    │      optimizer.py                          # Optimizer Module
    ├─data                                       # load data and data preprocessing
    │      env.py                                # Stablized config, not suitable to put in config.yaml, e.g., int and float type, constant and switch, etc.
    |      load_data.py                          # load different dataset
    │      load_inverse_data.py                  # load data for inverse problem
    │      load_multi_pde.py                     # load dataset containing multiple PDE forms (Forward Problem)
    │      load_single_pde.py                    # load dataset containing single PDE form (Forward Problem)
    │      pde_dag.py                            # Common Block for PDE DAG Construction and Graph-related Data
    │      utils_multi_pde.py                    # Dataset-related Blocks for Multiple PDE Forms (DAG Construction, PDE Form Generation, etc.)
    │      utils_dataload.py                     # Common Blocks for Data Loading
    └─utils                                      # Recording and Visualization Tools
            load_yaml.py                         # Load yaml file
            record.py                            # Record Experiment Results
            tools.py                             # Other Tools
            visual.py                            # Visualization Tools
```

## Model Execution

We provide configuration files for PDEformer models with varying numbers of parameters in the [configs/pretrain](configs/pretrain) folder. The corresponding checkpoints can be downloaded from [PKU Disk](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/C62E2F6D95624A79AB80EBD0AD9A7C1A) `pdeformer/ckpt`. The details are as follows:

| Model | Parameters | Configuration File | Checkpoint File |
| ---- | ---- | ---- | ---- |
| PDEformer-S | 3.60M | [configs/pretrain/pdeformer-S.yaml](configs/pretrain/pdeformer-S.yaml) | - |
| PDEformer-M | 6.92M | [configs/pretrain/pdeformer-M.yaml](configs/pretrain/pdeformer-M.yaml) | model-M_3M_pretrained.ckpt |
| PDEformer-L | 22.40M | [configs/pretrain/pdeformer-L.yaml](configs/pretrain/pdeformer-L.yaml) | model-L_3M_pretrained.ckpt |
| PDEformer-XL | 58.24M | [configs/pretrain/pdeformer-XL.yaml](configs/pretrain/pdeformer-XL.yaml) | model-XL_3M_pretrained.ckpt |

### Inference Example

The example code below demonstrates how to use PDEformer to predict the solution of a given PDE.
Before running, it is necessary to download the pre-trained PDEformer weights `pdeformer/ckpt/model-L_3M_pretrained.ckpt` from [PKU Disk](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/C62E2F6D95624A79AB80EBD0AD9A7C1A),
and change the value of the `model/load_ckpt` parameter in [configs/inference/pdeformer-L.yaml](configs/inference/pdeformer-L.yaml) to the path of the corresponding weight file.

```python
import numpy as np
import matplotlib.pyplot as plt
from mindspore import context
from mindspore import dtype as mstype
from src import load_config, get_model, PDENodesCollector, inference_pde

# Basic Settings
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
config, _ = load_config("configs/inference/pdeformer-L.yaml")
model = get_model(config, compute_type=mstype.float32)

# Define time-space coordinates
x_coord = np.linspace(-1, 1, 257)[:-1]
t_coord = np.linspace(0, 1, 101)

# solve PDE
pde = PDENodesCollector()
u = pde.new_uf()
pde.set_ic(u, x_coord, np.sin(np.pi * x_coord))
pde.sum_eq0(pde.dt(u), pde.dx(0.5 * pde.square(u)))

# Predict the solution with PDEformer and plot
pde_dag = pde.gen_dag(config)
u_pred = inference_pde(model, pde_dag, t_coord, x_coord)
plt.imshow(u_pred)
plt.show()
```

For more examples, please refer to the [PDEformer_inference_CN.ipynb](PDEformer_inference_CN.ipynb) notebook.

### Pre-training

**Generating Pre-training Dataset**

In order to pre-train the model effectively, the primary task is to prepare the pre-training data.
We use the traditional spectral numerical solver [Dedalus](https://dedalus-project.org/) to generate this pre-training data.
The pre-training dataset consists of solutions to (single-component) PDEs of the following forms:

$$
\begin{split}
u_t+f_0(u)+s(x)+(f_1(u)-\kappa(x)u_x)_x&=0 , \quad (t,x) \in [0,1] \times [-1,1], \\
u(0,x) &= g(x), \quad x \in [-1,1]
\end{split}
$$

where $f_i(u) = c_{i1}u+c_{i2}u^2+c_{i3}u^3$, $i=0,1$.
Each coefficient $c_{ik}$ is set to zero with a probability of $0.5$ (thus the corresponding term may not appear in the PDE's computational graph), otherwise, it is randomly drawn from $U([-3,3])$.
Coefficient fields $s(x), \kappa(x)$ are probabilistically set to be constant (no spatial dependence) or zero, where the range of values for $\kappa(x)$ is $[10^{-3},1]$.
The generation method for initial conditions $g(x)$ and non-constant source terms $s(x)$ follows the same approach as the initial condition generation in the [PDEBench dataset](https://arxiv.org/abs/2210.07182).
For non-periodic boundary conditions, the type of boundary conditions is randomly chosen from Dirichlet, Neumann, and Robin, and whether it is homogeneous is also chosen randomly, with the boundary conditions at the left and right endpoints generated independently.
For instructions on running the pre-training dataset generation code, please refer to [data_generation/README.md](data_generation/README.md).
Alternatively, you can download the dataset that we have generated, as described in [docs/DATASET.md](docs/DATASET.md).

#### Pre-training the Model

To pre-train the PDEformer-L model, we first need to adjust the configuration file [configs/pretrain/pdeformer-L.yaml](configs/pretrain/pdeformer-L.yaml).
In this configuration file, we need to specify the file path and filename of the dataset (without the `.hdf5` suffix):

```yaml
# ...
data:
    path: ../data_download  # dataset path
    # ...
    multi_pde:
        train:  # training dataset
            sinus0_c:  # dataset - periodic boundary conditions
                - custom_v4.21_sinus0_circ_cU3_k1e-03_1_seed2
                - custom_v4.21_sinus0_circ_cU3_k1e-03_1_seed3
                - custom_v4.21_sinus0_circ_cU3_k1e-03_1_seed4
                - custom_v4.21_sinus0_circ_cU3_k1e-03_1_seed5
                - custom_v4.21_sinus0_circ_cU3_k1e-03_1_seed6
                - ...
            sinus0_r:  # dataset - Robin boundary conditions
                - custom_v4.23_sinus0_robin_cU3_k1e-03_1_seed2
                - custom_v4.23_sinus0_robin_cU3_k1e-03_1_seed3
                - custom_v4.23_sinus0_robin_cU3_k1e-03_1_seed4
                - custom_v4.23_sinus0_robin_cU3_k1e-03_1_seed5
                - custom_v4.23_sinus0_robin_cU3_k1e-03_1_seed6
                # ...
        test:  # test dataset
            sinus0_c:
                - custom_v4.21_sinus0_circ_cU3_k1e-03_1_seed1
            sinus0_r:
                - custom_v4.23_sinus0_robin_cU3_k1e-03_1_seed1
# ...
```

The dataset files used in the example can be generated using [data_generation/custom_sinus.py](data_generation/custom_sinus.py), or can be downloaded from [PKU Disk](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/434EE9473D90449A8B1E4847065BCA89)  `pdeformer/sinus0`.
After completing the modifications to the configuration file, we can initiate single-machine, 8-card parallel training by running the following command:

```bash
# path to the config file
config_path=configs/pretrain/pdeformer-L.yaml

# preprocess data
python preprocess_data.py --config_file_path $config_path

# pretrain model
mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout \
    python train.py --config_file_path $config_path
```

or through the following shell script:

```bash
bash scripts/run_distributed_train.sh
```

The training logs, model weights, and files for visualization of experimental results of the pre-trained model will be saved in the `exp/pdeformer-L` directory.

### PDEBench Inference and Fine-tuning

The pre-trained PDEformer has shown exceptional generality in handling various equations. To assess its performance on forward problems, we can select some 1D equations from the PDEBench dataset, including the Burgers equation, the Advection equation, and the Reaction-Diffusion equation. Although our model can directly solve these equations (zero-shot inference), to achieve higher solving accuracy for specific types of equations, we can opt to further fine-tune the model.

To fine-tune with a specific PDEBench dataset (here using the Burgers equation as an example), we need to download the pretrained PDEformer weights `pdeformer/ckpt/model-L_3M_pretrained.ckpt` from [PKU Disk](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/C62E2F6D95624A79AB80EBD0AD9A7C1A), and download [1D_Burgers_Sols_Nu0.1.hdf5](https://darus.uni-stuttgart.de/api/access/datafile/133139) from the PDEBench dataset, then adjust the configuration file [configs/finetune/burgers-nu-0.1_pdeformer-L.yaml](configs/finetune/burgers-nu-0.1_pdeformer-L.yaml). In this configuration file, we need to specify the file path and filename of the dataset, as well as the path to the pre-trained model weights:

```yaml
# ...
model:
    # ...
    load_ckpt: ./exp/pdeformer-L/last_model.pth  # pretrained model weights
# ...
data:
    path: ../data_download  # dataset path
    num_samples_per_file:
        train: 9000  # number of samples in each training dataset file
        test: 1000  # number of samples in each test dataset file
    single_pde:
        param_name: burgers_nu2  # parameter name
        train: [0.1]  # viscosity
        test: [0.1]  # viscosity
    # ...
# ...
```

After completing the modifications to the configuration file, we can initiate a single-machine, single-card fine-tuning task by running the following command:

```bash
# path to the config file
config_path=configs/finetune/burgers-nu-0.1_pdeformer-L.yaml

# finetune model
python train.py --config_file_path $config_path --no_distributed --device_id 0
```

or through the following shell script:

```bash
bash scripts/run_standalone_train.sh
```

To fine-tune with other datasets from PDEBench, please refer to [docs/DATASET_CN.md](docs/DATASET_CN.md).

### Inverse Problem

#### Inverse Problem on Scalar Coefficients

In addition to solving forward problems, we can also use the pre-trained PDEformer to tackle the inverse problem of equation coefficient estimation.
For each PDE, we input the currently estimated coefficients into the pre-trained PDEformer to produce a predicted solution and obtain the recovered coefficients by minimizing the relative $L^2$ error with respect to the observational data.
Given that this optimization problem has many local minima, we employ the Particle Swarm Optimization (PSO) algorithm for the solution.

First, we need to download the pretrained [PDEformer weights](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/C62E2F6D95624A79AB80EBD0AD9A7C1A) `pdeformer/ckpt/model-L_3M_pretrained.ckpt` from PKU Disk, as well as the [inverse problem dataset](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/CDBCBEF0F0D4459C893F3CBBE62F521E) `pdeformer/inverse`. The datasets used for inverting scalars (equation coefficients) and inverting functions (source terms) are the same. For more detailed information about the inverse problem dataset, please refer to docs/DATASET_CN.md. Afterwards, adjust the configuration file [configs/inverse/inverse_scalar.yaml](configs/inverse/inverse_scalar.yaml). In this configuration file, we need to specify the file path and filename of the inverse problem dataset, as well as the path to the pretrained model weights:

```yaml
# ...
model:
    # ...
    load_ckpt: ./exp/pdeformer-L/last_model.pth  # pretrain model weights (path)
data:
    path: ../data_download  # Inverse problem dataset path
    # ...
# ...
inverse:
    data_file: custom_v4.23_inv_sinus0_circ_fS_cU3_k1e-03_1_seed1  # Inverse problem dataset file name
    system_identification: False  # Whether to perform system identification
    # ...
# ...
```

After completing the modifications to the configuration file, we can initiate a single-machine, single-card task for the inversion of equation coefficients by running the following command:

```bash
# path to the config file
config_path=configs/inverse/inverse_scalar.yaml

# run the inverse process
python inverse_scalar.py --config_file_path $config_path --device_id 0
```

or through the following shell script:

```bash
bash scripts/run_inverse_scalar.sh
```

#### Inverse Problem on Scalar Functions (Source Term)

We can also use the pre-trained PDEformer to address the function inversion problem (here, the function refers to the source term).
For each PDE, we set the current source term to be estimated as a trainable parameter, and input it into the pre-trained PDEformer to produce a predicted solution. We then use a gradient descent algorithm to optimize the source term parameters, aiming to minimize the relative $L^2$ error with respect to the observational data.

First, we need to download the pretrained [PDEformer weights](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/C62E2F6D95624A79AB80EBD0AD9A7C1A) `pdeformer/ckpt/model-L_3M_pretrained.ckpt` from PKU Disk, as well as the [inverse problem dataset](https://disk.pku.edu.cn/anyshare/zh-cn/link/AA6ABF7FEB034446069108D0B6B3920C35/768396ABFD014CF8B81FA886E6577D23/44E9E35239854ED8817718DFCCEA3B4D/CDBCBEF0F0D4459C893F3CBBE62F521E) `pdeformer/inverse`. The datasets used for inverting scalars (equation coefficients) and inverting functions (source terms) are the same. Afterwards, adjust the configuration file [configs/inverse/inverse_function.yaml](configs/inverse/inverse_function.yaml). In this configuration file, we need to specify the file path and filename of the inverse problem dataset, as well as the path to the pretrained model weights:

```yaml
# ...
model:
    # ...
    load_ckpt: ./exp/pdeformer-L/last_model.pth  # Path to pre-trained model weights
data:
    path: ../data_download  # Directory containing the inverse problem dataset
    # ...
# ...
inverse:
    data_file: custom_v4.23_inv_sinus0_circ_fS_cU3_k1e-03_1_seed1  # Inverse problem data file name (without suffix)
    # ...
# ...
```

After you've made the necessary modifications to the configuration file, you can start a single-machine, single-card task for function inversion by executing the following command in your terminal or command prompt:

```bash
# path to the config file
config_path=configs/inverse/inverse_function.yaml

# run the inverse process
python inverse_function.py --config_file_path $config_path --device_id 0
```

or through the following shell script:

```bash
bash scripts/run_inverse_function.sh
```

## References

* [Ye Z, Huang X, Chen L, et al. PDEformer: Towards a Foundation Model for One-Dimensional Partial Differential Equations[J]. arXiv preprint arXiv:2402.12652, 2024.](https://arxiv.org/abs/2402.12652)

* [Ying C, Cai T, Luo S, et al. Do transformers really perform badly for graph representation?[J]. Advances in neural information processing systems, 2021, 34: 28877-28888.](https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html)

* [Takamoto M, Praditia T, Leiteritz R, et al. PDEBench: An extensive benchmark for scientific machine learning[J]. Advances in Neural Information Processing Systems, 2022, 35: 1596-1611.](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html)

## Contributors

gitee id: functoreality, huangxiang360729, ChenLeheng, BingyangWu-pkusms21, juste_une_photo

email: yezhanhong@pku.edu.cn, sahx@mail.ustc.edu.cn, chenlh@pku.edu.cn, wby2003@stu.pku.edu.cn, ziningliu31@outlook.com
