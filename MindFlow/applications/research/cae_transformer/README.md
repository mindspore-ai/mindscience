# CAE-Transformer flow field prediction model

## Introduction

In order to effectively reduce the design cost and cycle time of using CFD methods, the reduced-order model (ROM) has gained wide attention in recent years. For complex compressible flows, using linear methods such as Proper Orthogonal Decomposition (POD) for flow field dimensionality reduction requires a large number of modes to ensure the reconstruction accuracy. It has been shown that the modes number can be effectively reduced by using nonlinear dimensionality reduction methods. Convolutional Autoencoder (CAE) is a kind of neural network composed of encoder and decoder, which can realize data dimensionality reduction and recon-struction, and can be regarded as a nonlinear extension of POD method. CAE is used for nonlinear dimension-ality reduction, and Transformer is used for time evolution. The CAE-Transformer can obtain high reconstruction and prediction accuracy on the premise of using less latents for unsteady compressible flows.

### Framework of CAE-Transformer

The basic framework of CAE-Transformer is mainly based on [paper1](https://doi.org/10.13700/j.bh.1001-5965.2022.0085) and [paper2](https://doi.org/10.1609/aaai.v35i12.17325). It consists of CAE and Transformer, where the encoder in CAE reduces the dimensionality of the time series flow field to achieve feature extraction, Transformer learns low dimensional spatiotemporal features and makes predictions, and the decoder in CAE realizes flow field reconstruction.

+ Input：Input the flow field for a period of time;

+ Compression：Extract high-dimensional spatiotemporal flow characteristics by dimensionality reduction of the flow field using the encoder of CAE;

+ Evolution：Learning the evolution of spatiotemporal characteristics of low dimensional spatial flow fields through Transformer and predicting the next moment;

+ Reconstruction：Restore the predicted low-dimensional features of the flow field to high-dimensional space through the decoder of CAE;

+ Output：Output the predicted results of the transient flow field at the next moment.

![CAE-Transformer.png](./images/cae_transformer_structure.png)

### Dataset

**Source**: Numerical simulation flow field data of two-dimensional flow around a cylinder, provided by the team of Associate Professor Yu Jian, School of Aeronautical Science and Engineering, Beijing University of Aeronautics and Astronautics.

**Format**: The data set is numerically simulated for the flow around a cylinder with 10 Reynolds numbers. The flow field data at each Reynolds number contains 401 time steps, and the flow field data at each time step is a 256*256 two-dimensional flow field. The data type of each variable is float32, and the total size of the dataset is about 1.96GB.

**Link**: [2D_cylinder_flow.npy](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-transformer/2D_cylinder_flow.npy)

## Quick Start

### Training method 1: Call the 'train.py' script in the command line

The model is trained by single machine and single card. According to the training task requirements, run train.py to start training; Before training, relevant training conditions need to be set in config.yaml.

`python -u train.py --mode GRAPH --save_graphs_path ./graphs --device_target GPU --device_id 0 --config_file_path ./config.yaml`

Among them,

`--mode` indicates the running mode, 'GRAPH' represents the static graph mode, 'PYNATIVE' represents the dynamic graph mode, and the default value is 'GRAPH';

`--device_target` indicates the type of computing platform used, you can choose 'Ascend' or 'GPU', the default value is 'GPU';

`--device_id` indicates the number of the calculated card used, which can be filled in according to the actual situation, and the default value is 0;

`--config_file_path` indicates the path to the parameter file, default value './config.yaml'.

### Training Method 2: Run Jupyter Notebook

You can use the Chinese and English versions of Jupyter Notebook to run training and verification code line by line:

Chinese version: [train_CN.ipynb](./cae_transformer_CN.ipynb)

English version: [train.ipynb](./cae_transformer.ipynb)

## Visualization of prediction results

Run the eval.py for post-processing operation, this operation will predict the dimensionality reduction and reconstruction data of CAE, the evolution data of Transformer, and the flow field data predicted by CAE-Transformer based on the weight parameter file of the training results.

`python -u eval.py`

The default output path for the above post-processing is `./prediction_result`, the save path can be modified in config.yaml.

## Prediction result

The following is a comparison of CAE-Transformer and the real value:

<figure class="harf">
    <img src="./images/prediction_result.gif" title="prediction result" width="500"/>
</figure>

The results show the velocity of different locations in the flow field over time. The average relative error between the predicted results and the true values is 6.3e-06.

## Performance

|        Parameter         |        NPU               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     Ascend, Memory32G   |      NVIDIA V100, Memory32G       |
|     MindSpore version   |        2.0.0             |      2.0.0       |
| Dataset | [Cylinder_flow](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-transformer/2D_cylinder_flow.npy) | [Cylinder_flow](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-transformer/2D_cylinder_flow.npy) |
|  Parameters | 3.8e5 | 3.8e5 |
|  Training hyperparameters | batch_size=32, epochs=100 | batch_size=32, epochs=100 |
|  Testing hyperparameters | batch_size=32 | batch_size=32 |
|  Optimizer | Adam | Adam |
|        Train loss      |        1.21e-6               |       1.21e-6      |
|        Validation loss      |        3.85e-7              |       3.86e-7       |
|        Speed          |     216ms/step        |    220ms/step  |

## Contributor

contributor's gitee id: [Marc-Antoine-6258](https://gitee.com/Marc-Antoine-6258)
contributor's email: 775493010@qq.com
