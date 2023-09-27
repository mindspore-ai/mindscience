ENGLISH | [简体中文](README_CN.md)

# Physics-encoded Recurrent Convolutional Neural Network (PeRCNN)

## Overview

PDE equations occupy an important position in the modeling of physical systems. But many underlying PDEs have not yet been fully explored in epidemiology, meteorological science, fluid mechanics, and biology. However, for those known PDE equations, such as Naiver-Stokes equations, the exact numerical calculation of these equations requires huge computing power, which hinders the application of numerical simulation in large-scale systems. Recently, advances in machine learning provide a new way for PDE solution and inversion.

![PeRCNN](images/percnn.jpg)

Recently, Huawei and Professor Sun Hao's team from Renmin University of China proposed Physics-encoded Recurrent Convolutional Neural Network, PeRCNN(https://www.nature.com/articles/s42256-023-00685-7) based on Ascend platform and MindSpore. Compared with physical information neural network, ConvLSTM, PDE-NET and other methods, generalization and noise resistance of PeRCNN are significantly improved. The long-term prediction accuracy is improved by more than 10 times. This method has broad application prospects in aerospace, shipbuilding, weather forecasting and other fields. The results have been published in nature machine intelligence.

The physical structure is embedded in PeRCNN. A π-convolution module combines partial physical prior and achieves nonlinear approximation by producting elements between feature graphs. This physical embedding mechanism guarantees that the model strictly follows a given physical equation based on our prior knowledge. The proposed method can be applied to various problems related with PDE systems, including data-driven modeling and PDE discovery, and can ensure accuracy and generality.。

## Samples

- [2d Burgers equation](./burgers_2d/)

- [3d Gray-Scott reaction-diffution equation](./gsrd_3d/)