ENGLISH | [简体中文](README_CN.md)

# Reduced order model for three-dimensional unsteady flow

## Overview

### Background

Three-dimensional complex flows are prevalent in practical engineering problems, and effectively capturing and analyzing the flow field poses significant challenges. The CFD simulation of 3D flow fields involving complex geometries demands substantial computing resources due to the increased mesh degrees of freedom. Consequently, this limitation hinders the progress of downstream tasks like interactive design and real-time optimal control.

While there has been extensive exploration of deep learning technology for flow field modeling in recent years, most of the focus has remained on two-dimensional shapes. As a result, there is still a noticeable gap when applying these models to real-world engineering scenarios. The stronger spatial coupling effect in 3D data compared to 2D data is a primary reason for this disparity. Additionally, training neural networks with a large number of model parameters requires robust computing power and ample storage resources.

For 3D unsteady flow, the reduced-order model based on the fully convolutional neural network called "ResUnet3D" can quickly establish the nonlinear mapping between snapshots of the 3D flow field, offering a promising approach to tackle these challenges.

### Model structure

The proposed neural network model follows the paradigm of an encoder-decoder architecture, which exhibits a symmetrical U-shaped structure. The main difference from the traditional Unet3D lies in the replacement of traditional convolutions with convolutional residual blocks.

![ResUnet3D.jpg](./images/ResUnet3D.jpg)

+ **Encoder**: The left side of the network, known as the contracting path, is responsible for hierarchically extracting the latent features of the high-dimensional flow field. The encoder consists of four downsampled residual blocks, as illustrated in Fig(a). Downsampling is accomplished by utilizing convolutional operations with a stride of 2 instead of pooling operations. Following each residual block operation, the number of feature channels is doubled while the size of the feature map is halved.

+ **Decoder**: On the right side of the network, referred to as the expansive path, the low-dimensional features are upsampled. Correspondingly, the decoding part also includes four upsampling residual blocks, with the structure of the upsampling residual block shown in Fig(b). The first step involves the application of deconvolution to increase the size of the original features by a factor of two while reducing the number of feature channels. It should be noted that the upsampled output block(c) responsible for the final output of the model discards the identity connection part in the upsampling residual block,.

+ **Residual Connect**: In addition to the residual connections within the residual blocks, we also introduced skip connections in our model, indicated by solid gray arrows in model architecture. The increased number of residual connections helps in capturing low-frequency features of the high-dimensional flow field, further enriching the details of the flow field prediction.

<div align="center">
  <img src='./images/blocks.jpg' width="500">
</div>

### Dataset

+ Source: Numerical simulation data of 3D flow around sphere, provided by Professor Chen Gang, School of Aerospace Engineering, Xi'an Jiaotong University

+ Establishment method: The calculation status and establishment method of the dataset can be found in [paper](https://arxiv.org/abs/2307.07323), and the case is an example in the paper based on Cartesian uniform sampling method

+ Data description:

    + Single flow state: Reynolds number Re=300

    + 400 consecutive moments of 3D flow snapshots, including non-dimensional pressure and different non-dimensional velocities in flow, normal and span directions

    + [Downloading link](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/3d_unsteady_flow)

+ Data directory structure

    ```text
        .
        └─data
        ├── original_data.npy        // raw flow snapshot
        ├── train_data.npy           // data for model train
        ├── eval_data.npy            // data for model evaluate
        ├── infer_data.npy           // data for model inference
        ├── train_data_norm.npy      // normalized data for model train
        ├── eval_data_norm.npy       // normalized data for model evaluate
        ├── infer_data_norm.npy      // normalized data for model inference
    ```

## QuickStart

### Run Option 1: Call `train.py` and `eval.py` from command line to start model training and  model inference, respectively

+ Example for training

```shell
# indirect mode
python train.py --config_file_path ./config.yaml --norm False --residual True --scale 1000.0 --mode GRAPH --device_target GPU --device_id 0

# direct mode
python train.py --config_file_path ./config.yaml --norm True --residual False --scale 1.0 --mode GRAPH --device_target GPU --device_id 0
```

where:

`--config_file_path` indicates the path of the parameter file. Default './config.yaml'.

`--norm` indicates whether the original data is normalized. Default: `False`

`--residual` indicates the forecast mode, `True` indicates the indirect (incremental) forecast mode, `False` indicates the direct forecast mode. Default: `True`

`--scale` indicates the scaling factor of the target. When making direct predictions (residual=False), scale=1, meaning the flow field in the future remains unscaled. However, for indirect predictions (residual=False), it is often necessary to enlarge the incremental flow field by a factor of scale=100.0 or 1000.0 times.

`--mode` is the running mode. 'GRAPH' indicates static graph mode. 'PYNATIVE' indicates dynamic graph mode. You can refer to [MindSpore official website](https://www.mindspore.cn/docs/zh-CN/master/design/dynamic_graph_and_static_graph.html) for details.Default 'GRAPH'.

`--device_target` indicates the computing platform. You can choose 'Ascend' or 'GPU'. Default 'Ascend'.

`--device_id` indicates the index of NPU or GPU. Default 0.

For other quick start parameter configurations, please refer to `default_config.yaml`.

+ Example for inference

After training the model, select the optimal model for iterative reasoning. The primary purpose of the parameters is to specify the path where the model parameters(`.ckpt`) are stored.

```shell
python eval.py --config_file_path ./config.yaml --norm False --residual True --scale 1000.0 --device_target GPU --device_id 0
```

The parameter configuration is consistent with the training script `train.py`, please refer to `default_config.yaml`.

### Run Option 2: Run Jupyter Notebook

You can use [Chinese](flow_around_sphere_CN.ipynb) or [English](flow_around_sphere.ipynb) Jupyter Notebook to run the training and evaluation code line-by-line.

## Results

Compared with traditional CFD technology, the trained neural network significantly enhances the efficiency of high-fidelity flow field predictions, capable of generating a period of flow field data within a few seconds. The subsequent partial reasoning flow field results based on indirect prediction are consistent with the CFD results, with a maximum relative error of only 0.002. Moreover, the maximum error is concentrated in the wake area.

+ Pressure dynamic cloud maps on the section `Z=0`:

<div align="center">
  <img src="./images/P.gif"  width="800"/>
</div>

+ Dynamic cloud maps of streamwise velocity on section `Z=0`:

<div align="center">
  <img src="./images/U.gif" width="800"/>
</div>

The figures clearly show noticeable periodic characteristics, with various flow physical quantities maintaining a relatively stable state as time progresses. This indicates that the model is capable of achieving long-term stable predictions.

+ In order to show the three-dimensional flow characteristics more clearly, the three-dimensional vorticity isosurface (Q_criterion=0.0005）colored by the velocity field after two cycles is as follows:

<div align="center">
  <img src="./images/Q.png" width="600"/>
</div>

The model's prediction of the hairpin wake vortex is comparable to the CFD results, with similarities observed. However, there are significant fluctuations in the vortex's tail, indicating the model's capability to extract features related to the flow field's spatial structure. Nevertheless, there is still room for improvement in the model's performance.

## Contributor

gitee id: [lixin](https://gitee.com/lixin07)

email: lixinacl@stu.xjtu.edu.cn