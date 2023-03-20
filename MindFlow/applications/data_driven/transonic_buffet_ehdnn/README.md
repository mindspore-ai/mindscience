---

# Introduction

At transonic flow conditions, the self-sustained large-scale oscillation of shock wave on airfoils is called transonic buffet. The reason is related to the flow separation and interaction between shock wave and boundary layer. After entering the buffet boundary, the change of separation zone induces the flow instability and affects the position of shock wave, which makes the shock wave move forward and backward and comes with complex unsteady and nonlinear characteristics. Learning the unsteady shock buffet flow directly from the flow field (spatial-temporal flow characteristics) is a valuable and challenging method for buffet research. In order to find an efficient DL modeling method for the complex unsteady transonic buffet flow, an enhanced hybrid deep neural network (eHDNN) is designed to predict the unsteady flow field based on flow field reconstruction.
<img src="https://i.postimg.cc/Xqp3TKQj/p1.png" title="p1.png" alt="" data-align="center">

# Framework of eHDNN

The basic framework of the eHDNN is mainly based on the hybrid deep neural network framework proposed by the previous work [paper](https://doi.org/10.1016/j.ast.2022.107636), which is constituted by CNN, ConvLSTM and DeCNN.CNN reduces the dimensionality of time series flow fields and realizes the characteristic extraction; ConvLSTM learns the evolution of low-dimensional spatial-temporal characteristics and make prediction; finally, DeCNN achieves the reconstruction of predicted flow field characteristics.

+ Input layer: inputting the historical flow fields;
+ Convolutional layer: reducing the dimensionality of flow fields and extract the high-dimensional spatial-temporal flow characteristics by CNN;
+ Memory layer: learning the evolution of the spatial-temporal characteristics of flow fields in the low-dimensional space and predicting the next moment by ConvLSTM;
+ Deconvolutional output layer: restoring the predicted low-dimensional characteristics of flow fields to high-dimensional space to achieve the reconstruction of the prediction of transient flow field at the next moment by DeCNN, then outputting the prediction.

![](https://i.postimg.cc/d3pvv7x8/p2.jpg)

# Training samples

+ Source：Numerical simulation flow field data of OAT15A supercritical airfoil unsteady buffet were calculated and provided by Professor Wang Gang's team at School of Astronautics, Northwestern Polytechnical University

+ Establishment method: See the [paper]() for the calculation state and establishment method of data set

+ Specification: Single-state data sets and multi-state data sets

    + Single-state: Instantaneous snapshots of buffet flow field at 9200 time steps under a single angle of attack (7.8g)
    + Multi-state：The buffet flow field transient snapshots of 6 groups of states with 9200 time steps(the angle of attack range of 3.3° to 3.8° (varying with 0.1° interval))
    + Note: Each flow field snapshot contains 3 channels, representing pressure distribution information, chord velocity information and normal velocity information of the flow field
    + The download address of the data samples is [https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/airfoil/2D_unsteady/](https://gitee.com/link?target=https%3A%2F%2Fdownload.mindspore.cn%2Fmindscience%2Fmindflow%2Fdataset%2Fapplications%2Fdata_driven%2Fairfoil%2F2D_unsteady%2F)

# Training process

The model is trained by single machine and single card. According to the training task requirements, run train.py to start training:
Before training, relevant training conditions need to be set in config.yaml:

+ Correlation path setting
+ Training parameters setting

# Visualization of prediction results

Run prediction.py according to the training conditions
Post-processing operation:

+ The prediction data is post-processed, and the contour snapshots of the predicted flow field are output (Flow field data format is Tecplot format, open with Tecplot to view the results)

+ The default saving path is prediction_result. The path contains the CFD flow field snapshot of the specified flow field variables, the predicted flow field snapshots, and the absolute error contours

+ Visualization：

  1. Open Tecplot software, import Tecplot format data, and select predicted flow field in batches (sequence flow field length T=84)
  2. Select Tecplot Data Loader as the loading data format

# Prediction result

The following figure shows the prediction results of unsteady buffet flow fields in a single period under the angle of attack of 3.75° (generalized state) based on the well-trained eHDNN model (pressure field).

<figure class="harf">
    <img src="./images/375_pressure_cfd.gif" title="cfd" width="200"/>
    <img src="./images/375_pressure_prediction.gif" title="prediction" width="200"/>
    <img src="./images/375_pressure_abserror.gif" title="abs error" width="200"/>
</center>
