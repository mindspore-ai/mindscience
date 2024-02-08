# Multi-timestep Complicated Flow Field Prediction with Transonic Airfoil under Data Driven Mode(with Two Kinds of Backbones: FNO2D and UNET2D)

## Background

High precision unsteady flow simulation is among the key topics in computational fluid dynamics(CFD), with a wide range of application scenarios and broad market value. However, traditional methods encountered problems such as long time-consumption, poor precision and hard convergence. AI methods provides a new perspective to explore the evolution mechanism of flow fields.

The current application provides an end-to-end solution for predicting unsteady and complex flow fields in a two-dimensional transonic airfoil scenario. Two network backbones, Fourier Neural Operator (FNO) and Unet, are constructed in order to stably predict the flow field for subsequent *m* time steps based on the input flow field of *k* time steps, while ensuring a certain level of accuracy. It can verify the effectiveness of deep learning methods in predicting unsteady flow fields under multiple physical parameter changes in complex flow structures such as shock waves.

$$
u_{[t_0\sim t_{k-1}]} \mapsto u_{[t_k\sim t_{k+m}]}
$$

![Fig1](images/img_1.PNG)
<center>Fig1. Flow field streamwise velocity comparison under different cases((a-d)：CFD results；(e-h)：FNO results；(i-l)Unet results</center>

## Technical Path

The application of airfoil2D-unsteady consists of two parts, namely **model and dataset preparation** part and **core model** part.

### Model and Dataset Preparation

The dataset of current application has a dimension of 4, arranged in *THWC* sequence, where channel(C) includes streamwise velocity *U*, circumferential velocity *V* and pressure *P*. When preparing data, *T_in* timesteps are merged as inputs to the core model. Thus the input size of the dataset is (*B, T_in, H, W, C*), and the label size of the dataset is (*B, T_out, H, W, C*).  Traversing of *T_ out* timesteps is needed so that the label size of each step dataset is (*B, H, W, C*) and the dataset input is merged into (*B \* T_in, H, W, C*). The input and label are updated after training for one step until traversing of of *T_ out* timesteps ends.

### Core Model

- FNO2D

   FNO2D structure is presented in the following figure. *P* and *Q* are all fully connection layers, where *P* is lifting Layer to implement high-dimensional mapping of input vectors. The mapping results are inputs for Fourier layer，nesting several layers to perform nonlinear transformation of frequency domain information. Finally, *Q* provides a mapping to the final prediction from nesting results, acting as a decoding layer.

![Fig2](images\FNO.PNG)

- Unet2D

   Unet2D structure is presented in the following figure。It mainly consists of upsampling blocks and downsampling blocks. The downsampling block gradually reduces data dimensions and increases feature information through convolution and pooling operations. The upsampling block adds an deconvolution layer to gradually increase the data dimension and reduce feature information; At the same time, upsampling also includes skip connections, connecting the downsampling output with the corresponding upsampling output as input to the convolutional layer in upsampling.

![Fig3](images/Unet.PNG)

## QuickStart

Dataset download link：[data_driven/airfoil/2D_unsteady](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/airfoil/2D_unsteady/). Save the dataset to `./dataset`.

The following two training types are provided:

- Call `train.py` from command line

  ```python
  python train.py --config_path ./config/config.yaml --device_target Ascend --device_id 0 --mode GRAPH --backbone UNET2D

  ```

  where：

  --config_path indicates the path of the parameter file. Default "./config/config.yaml".

  --device_target indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default "Ascend".

  --device_id indicates the index of NPU or GPU. Default 0.

  --mode is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. Default "GRAPH"

  --backbone indicates the backbone of the core model. Default "UNET2D".

- Run Jupyter Notebook

  You can use[Chinese](./2D_unsteady_CN.ipynb) 和[English](./2D_unsteady.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

## Performance

|         Parameters         |        NPU         |           GPU           |
| :------------------------: | :----------------: | :---------------------: |
|          hardware          | Ascend(memory 32G) | NVIDIA V100(memory 32G) |
|     MindSpore version      |       2.0.0        |          2.0.0          |
|         T_in/T_out         |        8/32        |          8/32           |
|         data_size          |        3600        |          3600           |
|           epochs           |        200         |           200           |
|         optimizer          |  AdamWeightDecay   |     AdamWeightDecay     |
|   FNO2D train loss(RMSE)   |       6.9e-3       |         6.8e-3          |
| **FNO2D test loss(RMSE)**  |     **5.5e-3**     |       **5.4e-3**        |
|  **FNO2D speed(s/step)**   |      **0.68**      |        **1.07**         |
|  Unet2D train loss(RMSE)   |       5.8e-3       |         5.1e-3          |
| **Unet2D test loss(RMSE)** |     **5.1e-3**     |       **4.7e-3**        |
|  **Unet2D speed(s/step)**  |      **0.49**      |        **1.46**         |

## Contributor

gitee id: [mengqinghe0909](https://gitee.com/mengqinghe0909)
email: mengqinghe0909@126.com

## Reference

Deng Z, Liu H, Shi B, et al. Temporal predictions of periodic flows using a mesh transformation and deep learning-based strategy[J]. Aerospace Science and Technology, 2023, 134: 108081. https://doi.org/10.1016/j.ast.2022.108081