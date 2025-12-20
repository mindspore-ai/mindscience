ENGLISH | [CHINESE](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/fno2d/README_CN.md)

# Solve 2D Navier-Stokes Equation by FNO

## Background Introduction

### Overview

Computational fluid dynamics is one of the most important techniques in the field of fluid mechanics in the 21st century. The flow analysis, prediction and control can be realized by solving the governing equations of fluid mechanics by numerical method. Traditional finite element method (FEM) and finite difference method (FDM) are inefficient because of the complex simulation process (physical modeling, meshing, numerical discretization, iterative solution, etc.) and high computing costs. Therefore, it is necessary to improve the efficiency of fluid simulation with AI.

Machine learning methods provide a new paradigm for scientific computing by providing a fast solver similar to traditional methods. Classical neural networks learn mappings between finite dimensional spaces and can only learn solutions related to specific discretizations. Different from traditional neural networks, Fourier Neural Operator (FNO) is a new deep learning architecture that can learn mappings between infinite-dimensional function spaces. It directly learns mappings from arbitrary function parameters to solutions to solve a class of partial differential equations.  Therefore, it has a stronger generalization capability. More information can be found in the paper, [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895).

### Fourier Neural Operator

The architecture of the Fourier Neural Operator (FNO) model is shown in the following figure. wherein $w_0(x)$ denotes the initial vorticity. The Lifting Layer achieves high-dimensional mapping of the input vector, and the mapped result is then fed into the Fourier Layer for nonlinear transformation of frequency-domain information. Finally, the Decoding Layer maps the transformed result to the final prediction $w_1(x)$.

The Lifting Layer, Fourier Layer, and Decoding Layer collectively constitute the Fourier Neural Operator.

![Fourier Neural Operator network structure](https://raw.atomgit.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_driven/navier_stokes/fno2d/images/FNO.png)

The network structure of the Fourier Layer is illustrated in the following figure. In the figure, V represents the input vector. The upper block indicates that after the vector undergoes Fourier transform, it is subjected to linear transformation R to filter high-frequency information, followed by inverse Fourier transform. The other branch undergoes linear transformation W, and finally passes through an activation function to obtain the output vector of the Fourier Layer.

![Fourier Layer network structure](https://raw.atomgit.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_driven/navier_stokes/fno2d/images/FNO-2.png)

This tutorial introduces the solution method for the Navier-Stokes equation using the Fourier Neural Operator.

### Navier-Stokes equation

Navier-Stokes equation is a classical equation in computational fluid dynamics. It is a set of
partial differential equations describing the conservation of fluid momentum, called N-S equation
for short. Its vorticity form in two-dimensional incompressible flows is as follows:

$$
\partial_t w(x, t)+u(x, t) \cdot \nabla w(x, t)=\nu \Delta w(x, t)+f(x), \quad x \in(0,1)^2, t \in(0, T]
$$

$$
\nabla \cdot u(x, t)=0, \quad x \in(0,1)^2, t \in[0, T]
$$

$$
w(x, 0)=w_0(x), \quad x \in(0,1)^2
$$

where $u$ is the velocity field, $w=\nabla \times u$ is the vorticity, $w_0(x)$ is the initial
vorticity, $\nu$ is the viscosity coefficient, $f(x)$ is the forcing function.

We aim to solve two-dimensional incompressible N-S equation by learning the Fourier Operator mapping from each time step to the next time step:

$$
w_t \mapsto w(\cdot, t+1)
$$

## Model Implementation

### Hardware Requirements

NPU memory>32G

### MindSpore & MindScience Version

mindspore>=2.7.0
mindscience==0.8.0

### Installation

1. Ensure that the correct versions of MindSpore and MindScience are installed in the environment;
2. Clone the MindScience repository or obtain the codes directly from [MindFlow/applications/data_driven/navier_stokes/fno2d/](https://atomgit.com/mindspore-lab/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/fno2d);

### Dataset

Download train and test dataset: [data_driven/navier_stokes/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/dataset/).

### Coding

#### QuickStart

You can download dataset from [data_driven/navier_stokes/](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/). Save these dataset at `./dataset`.

#### Run Option 1: Call `train.py` from command line

You can download the training python scripts from [train.py](ttps://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/fno2d/train.py).

```shell
export PYTHONPATH=$(cd ../../../../../ && pwd):$PYTHONPATH
python train.py --config_file_path ./configs/fno2d.yaml --mode GRAPH --device_target Ascend --device_id 0
```

where:

`--config_file_path` indicates the path of the parameter file. Default './configs/fno2d.yaml';

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. Default 'GRAPH'.

`--device_target` indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default 'Ascend'.

`--device_id` indicates the index of NPU or GPU. Default 0.

#### Run Option 2: Run Jupyter Notebook

You can use [Chinese](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/fno2d/FNO2D_CN.ipynb) or [English](https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/fno2d/FNO2D.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

## Experiment Results Display

Take 1 samples, and do 10 consecutive steps of prediction. Visualize the prediction as follows.

![Inference Error](https://raw.atomgit.com/mindspore-lab/mindscience/raw/master/MindFlow/applications/data_driven/navier_stokes/fno2d/images/result.gif)

### Performance

| Parameter               | Ascend               |
|:----------------------:|:--------------------------:|
| Hardware                | Ascend 32G           |
| MindSpore version       | 2.7.0                |
| dataset                 | [2D Navier-Stokes Equation Dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)      |
| Parameters              | 9e5                  | 9e5                   |
| Train Config            | batch_size=19, steps_per_epoch=1000, epochs=150 |
| Optimizer               | Adam                 |
| Train Loss(MSE)         | 0.4                 |
| Evaluation Error(RMSE)  | 0.06                |
| Speed(ms/step)          | 32                   |

## License

- License: `Apache License 2.0`
- License Link: `https://atomgit.com/mindspore-lab/mindscience/blob/master/LICENSE`

## Acknowledgments

### Contributors

gitee id: [yi-zhang95](https://gitee.com/yi-zhang95),[huangwangwen2025](https://gitee.com/huangwangwen2025)

email: zhang_yi_1995@163.com,wangwen@isrc.iscas.ac.cn

## Contact Us

If you have any suggestions for MindScience, please contact us via [issue](https://atomgit.com/mindspore-lab/mindscience/issues), and we will address them promptly.

## Quotation

If this project is helpful to your research, please cite the relevant work.