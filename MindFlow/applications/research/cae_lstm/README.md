[简体中文](README_CN.md) | ENGLISH

# CAE-LSTM reduced-order model

## Overview

### Background

In order to effectively reduce the design cost and cycle time of using CFD methods, the reduced-order model (ROM) has gained wide attention in recent years. For complex compressible flows, using linear methods such as Proper Orthogonal Decomposition (POD) for flow field dimensionality reduction requires a large number of modes to ensure the reconstruction accuracy. It has been shown that the modes number can be effectively reduced by using nonlinear dimensionality reduction methods. Convolutional Autoencoder (CAE) is a kind of neural network composed of encoder and decoder, which can realize data dimensionality reduction and recon-struction, and can be regarded as a nonlinear extension of POD method. CAE is used for nonlinear dimension-ality reduction, and Long Short-Term Memory (LSTM) is used for time evolution. The CAE-LSTM can obtain high reconstruction and prediction accuracy on the premise of using less latents for unsteady compressible flows.

### Model structure

The basic framework of CAE-LSTM is mainly based on paper: [Ruoye Xiao, Jian Yu, Zhengxiao Ma. Applicability of Convolutional Autoencoder in reduced-order model of unsteady compressible flows[J/OL]. Journal of Beihang University: 1-16[2023-07-25].DOI:10.13700/j.bh.1001-5965.2022.0085.](https://doi.org/10.13700/j.bh.1001-5965.2022.0085). It consists of CAE and LSTM, where the encoder in CAE reduces the dimensionality of the time series flow field to achieve feature extraction, LSTM learns low dimensional spatiotemporal features and makes predictions, and the decoder in CAE realizes flow field reconstruction.

+ Input：Input the flow field for a period of time
+ Compression：Extract high-dimensional spatiotemporal flow characteristics by dimensionality reduction of the flow field using the encoder of CAE
+ Evolution：Learning the evolution of spatiotemporal characteristics of low dimensional spatial flow fields through LSTM and predicting the next moment
+ Reconstruction：Restore the predicted low-dimensional features of the flow field to high-dimensional space through the decoder of CAE
+ Output：Output the predicted results of the transient flow field at the next moment

![CAE-LSTM.png](./images/cae_lstm.png)

### Dataset

Source: Numerical simulation flow field data of one-dimensional Sod shock tube, Shu-Osher problem, Tow-dimensional Riemann problem, Kelvin-Helmholtz instability problem and cylinder flow, provided by Professor Yu Jian from the School of Aeronautic Science and Engineering, Beihang University

Establishment method: The calculation status and establishment method of the dataset in the first four cases can be found in paper: [Ruoye Xiao, Jian Yu, Zhengxiao Ma. Applicability of Convolutional Autoencoder in reduced-order model of unsteady compressible flows[J/OL]. Journal of Beihang University: 1-16[2023-07-25].DOI:10.13700/j.bh.1001-5965.2022.0085.](https://doi.org/10.13700/j.bh.1001-5965.2022.0085)
The cylinder case can be found in paper: [Ma Z, Yu J, Xiao R. Data-driven reduced order modeling for parametrized time-dependent flow problems[J]. Physics of Fluids, 2022, 34(7).](https://pubs.aip.org/aip/pof/article/34/7/075109/2847227/Data-driven-reduced-order-modeling-for)

Data description:
Sod shock tube: The coordinate range is \[0, 1\], and there is a thin film at x=0.5 in the middle. At the initial moment, remove the thin film in the middle of the shock tube and study the changes in gas density in the shock tube. The calculation time t ranges from \[0, 0.2\] and is divided into an average of 531 time steps. A total of 531 flow field snapshots, each with a matrix size of 256.

Shu-Osher problem: The coordinate range is \[-5, 5\], and the calculation time t ranges from \[0, 1.8] and is divided into an average of 2093 time steps. A total of 2093 flow field snapshots, each with a matrix size of 512.

Riemann problem: The coordinate range is \[0, 1\], and the calculation time t ranges from \[0, 0.25]. A total of 1250 flow field snapshots, each with a matrix size of (128, 128).

Kelvin-Helmholtz instability problem: The coordinate range is \[-0.5, 0.5\], and the calculation time t ranges from \[0, 1.5] and is divided into an average of 1786 time steps. A total of 1786 flow field snapshots, each with a matrix size of (256, 256).

cylinder flow: Using 128✖128 grids to interpolate flow field data for convolutional network processing. There are a total of 51 Reynolds (Re = 100, 110, 120, ..., 600), and there are 401 flow field snapshots in each Reynolds, each with a matrix size of (128, 128).

The download address for the dataset is: [data_driven/cae-lstm/dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm)

## QuickStart

### Run Option 1: Call `cae_train.py` and `lstm_train.py` from command line to start train cae and lstm network, respectively

+ Train the CAE network:

`python -u cae_train.py --case sod --mode GRAPH --device_target GPU --device_id 0 --config_file_path ./config.yaml`

+ Train the LSTM network:

`python -u lstm_train.py --case sod --mode GRAPH --device_target GPU --device_id 0 --config_file_path ./config.yaml`

where:
`--case` indicates the case to run. You can choose 'sod', 'shu_osher', riemann', 'kh' or 'cylinder'. Default 'sod'，where 'sod' and 'shu_osher' are one dimension cases, 'riemann', 'kh' and 'cylinder' are two dimension cases

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. You can refer to [MindSpore official website](https://www.mindspore.cn/docs/zh-CN/master/design/dynamic_graph_and_static_graph.html) for details.Default 'GRAPH'.

`--device_target` indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default 'Ascend'.

`--device_id` indicates the index of NPU or GPU. Default 0.

`--config_file_path` indicates the path of the parameter file. Default './config.yaml'.

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](./cae_lstm_CN.ipynb) or [English](./cae_lstm.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

## Results

The following are the actual flow field, CAE-LSTM prediction results, and prediction errors of the five cases.

The first two flow field results for each case show the variation (density for the first four cases and streamwise velocity for the cylinder case) in the flow field over time, while the third error curve shows the variation of the average error between the CAE-LSTM flow field and the real flow field label over time. The overall prediction time error meet the accuracy requirements of flow field prediction.

Sod shock tube:
<figure class="harf">
    <img src="./images/sod_cae_lstm_predict.gif" title="sod_cae_lstm_predict" width="500"/>
    <img src="./images/sod_cae_lstm_error.png" title="sod_cae_lstm_error" width="250"/>
</figure>

Shu-Osher problem:
<figure class="harf">
    <img src="./images/shu_osher_cae_lstm_predict.gif" title="shu_osher_cae_lstm_predict" width="500"/>
    <img src="./images/shu_osher_cae_lstm_error.png" title="shu_osher_cae_lstm_error" width="250"/>
</figure>

Riemann problem:
<figure class="harf">
    <img src="./images/riemann_cae_lstm_predict.gif" title="riemann_cae_lstm_predict" width="500"/>
    <img src="./images/riemann_cae_lstm_error.png" title="riemann_cae_lstm_error" width="250"/>
</figure>

Kelvin-Helmholtz instability problem:
<figure class="harf">
    <img src="./images/kh_cae_lstm_predict.gif" title="kh_cae_lstm_predict" width="500"/>
    <img src="./images/kh_cae_lstm_error.png" title="kh_cae_lstm_error" width="250"/>
</figure>

cylinder flow (Re = 200)：
<figure class="harf">
    <img src="./images/cylinder_cae_lstm_predict.gif" title="cylinder_cae_lstm_predict" width="500"/>
    <img src="./images/cylinder_cae_lstm_error.png" title="cylinder_cae_lstm_error" width="250"/>
</figure>

## Performance

Sod：

|        Parameter         |        NPU               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     Ascend: Memory32G       |      NVIDIA V100 cores 32G       |
|     MindSpore version   |        2.0.0         |2.0.0|
|     Dataset          |      [Sod](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/sod/)             |      [Sod](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/sod/)       |
|      Parameters       |       6e4       |         6e4         |
|      Training hyperparameters     |    cae_batch_size=8, lstm_batch_size=4, steps_per_epoch=1, epochs=4400 | cae_batch_size=8, lstm_batch_size=4, steps_per_epoch=1, epochs=4400 |
|     Optimizer         |        Adam     |        Adam         |
|     Train loss   |      5e-6(cae), 1e-3(lstm)        |     3e-6(cae), 5e-5(lstm)       |
|      Speed (ms/step)   |     320(cae), 1350(lstm)       |    400(cae), 800(lstm)  |

Shu-Osher：

|        Parameter         |        NPU               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     Ascend: Memory32G       |      NVIDIA V100 cores 32G       |
|     MindSpore version   |        2.0.0             |      2.0.0       |
|     Dataset          |      [Shu-Osher](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/sod/)             |      [Shu-Osher](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/sod/)       |
|      Parameters       |       6e4       |         6e4         |
|      Training hyperparameters     |    cae_batch_size=16, lstm_batch_size=16, steps_per_epoch=1, epochs=4400 | cae_batch_size=16, lstm_batch_size=16, steps_per_epoch=1, epochs=4400 |
|     Optimizer         |        Adam     |        Adam         |
|     Train loss    |      0.0015(cae), 0.001(lstm)        |     0.0015(cae), 0.0003(lstm)       |
|     Speed (ms/step)   |     900(cae), 7350(lstm)       |    750(cae), 4300(lstm)  |

Riemann：

|        Parameter         |        NPU               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     Ascend: Memory32G       |      NVIDIA V100 cores 32G       |
|     MindSpore version   |        2.0.0             |      2.0.0       |
|     Dataset          |      [Riemann](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/riemann/)             |      [Riemann](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/riemann/)       |
|      Parameters       |       6e4       |         6e4         |
|      Training hyperparameters     |    cae_batch_size=16, lstm_batch_size=32, steps_per_epoch=1, epochs=4400 | cae_batch_size=16, lstm_batch_size=32, steps_per_epoch=1, epochs=4400 |
|     Optimizer         |        Adam     |        Adam         |
|     Train loss    |      1e-4(cae), 5e-3(lstm)        |     5e-5(cae), 1e-4(lstm)       |
|     Speed (ms/step)   |     900(cae), 700(lstm)       |    1000(cae), 800(lstm)  |

KH：

|        Parameter         |        NPU               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     Ascend: Memory32G       |      NVIDIA V100 cores 32G       |
|     MindSpore version   |        2.0.0             |      2.0.0       |
|     Dataset          |      [KH](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/kh/)             |      [KH](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/kh/)       |
|      Parameters       |       6e4       |         6e4         |
|      Training hyperparameters     |    cae_batch_size=32, lstm_batch_size=32, steps_per_epoch=1, epochs=4400 | cae_batch_size=32, lstm_batch_size=32, steps_per_epoch=1, epochs=4400 |
|     Optimizer         |        Adam     |        Adam         |
|     Train loss    |      1e-3(cae), 5e-4(lstm)        |     1e-3(cae), 1e-5(lstm)       |
|     Speed (ms/step)   |     2000(cae), 1300(lstm)       |    2200(cae), 1500(lstm)  |

Cylinder（Re = 200）：

|        Parameter         |        NPU               |    GPU       |
|:----------------------:|:--------------------------:|:---------------:|
|     Hardware         |     Ascend: Memory32G       |      NVIDIA V100 cores 32G       |
|     MindSpore version   |        2.0.0             |      2.0.0       |
|     Dataset          |      [Cylinder](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/cylinder_flow/)             |      [Cylinder](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/cylinder_flow/)       |
|      Parameters       |       6e4       |         6e4         |
|      Training hyperparameters     |    cae_batch_size=8, lstm_batch_size=16, steps_per_epoch=1, epochs=4400 | cae_batch_size=8, lstm_batch_size=16, steps_per_epoch=1, epochs=4400 |
|     Optimizer         |        Adam     |        Adam         |
|     Train loss    |      1e-4(cae), 1e-4(lstm)        |     5e-5(cae), 1e-4(lstm)       |
|     Speed (ms/step)   |     500(cae), 200(lstm)       |    500(cae), 200(lstm)  |

## Contributor

gitee id: [xiaoruoye](https://gitee.com/xiaoruoye)

email: 1159053026@qq.com
