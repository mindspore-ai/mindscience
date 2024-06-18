# Solve Navier-Stokes Equation by SNO-3D

## Overview

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

We aim to learn the operator mapping the vorticity at the first 10 time steps to the full trajectory [10, T]:

$$
w|_{(0, 1)^2 \times [0, 10]} \mapsto w|_{(0, 1)^2 \times [10, T]}
$$

### Technical path

Spectral Neural Operator is the FNO-like architecture using polynomial transformation to spectral space (Chebyshev, Legendre, etc.) instead of Fourier.
To compute forward and inverse polynomial transformation matrices for spectral convolutions, the input should be interpolated at the respective Gauss quadrature nodes (Chebyshev grid, etc.).
The interpolated input is lifted to a higher dimension channel space by a convolutional Encoder layer. The result comes to the input of a sequence of spectral (SNO) layers, each of which applies a linear convolution to its truncated spectral representation. The output of SNO layers is projected back to the target dimension by a convolutional Decoder, and finally interpolated back to the original nodes.

The spectral (SNO) layer performs the following operations: applies the polynomial transformation $A$ to spectral space (Chebyshev, Legendre, etc.); a linear convolution $L$ on the lower polynomial modes and filters out the higher modes; then applies the inverse conversion $S={A}^{-1}$ (back to the physical space). Then a linear convolution $W$ of input is added, and nonlinear activation is applied.

![Spectral Neural Operator model structure](images/sno.png)

## QuickStart

You can download dataset from  [data_driven/navier_stokes3d/](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes_3d/). Save these dataset at `./dataset`.

### Run Option 1: Call `train.py` from command line

```shell
python train.py --config_file_path ./configs/sno3d.yaml --mode GRAPH --device_target Ascend --device_id 0
```

where:

`--config_file_path` indicates the path of the parameter file. Default './configs/fno3d.yaml'；

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. Default 'GRAPH'.

`--device_target` indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default 'Ascend'.

`--device_id` indicates the index of NPU or GPU. Default 0.

### Run Option 2: Run Jupyter Notebook

You can use [English](./SNO3D.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

## Results

Visualize the prediction for the first test sample as follows.

![Predict](./images/result.gif)

## Performance

| Parameter               | Ascend               | GPU                |
|:----------------------:|:--------------------------:|:---------------:|
| Hardware                | Ascend 64G           | NVIDIA V100 16G    |
| MindSpore version       | >=2.2.0                | >=2.2.0                   |
| dataset                 | [3D Navier-Stokes Equation Dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes_3d/)      | [3D Navier-Stokes Equation Dataset](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes_3d/)                   |
| Parameters              | 7e6                  | 7e6                    |
| Train Config            | batch_size=5, steps_per_epoch=200, epochs=120 | batch_size=5, steps_per_epoch=200, epochs=120 |
| Evaluation Config       | batch_size=1      | batch_size=1               |
| Optimizer               | AdamWeightDecay     | AdamWeightDecay              |
| Train Loss(MSE)         | 0.012                | 0.012             |
| Evaluation Error(RMSE)  | 0.13                | 0.13              |
| Speed(ms/step)          | 100                   | 200                 |

## Contributor

gitee id：[juliagurieva](https://gitee.com/JuliaGurieva)

email: gureva-yulya@list.ru
