ENGLISH | [简体中文](README.md)

# G-TEAM Earthquake Early Warning Model

## Overview

The earthquake early warning system aims to issue alerts before destructive seismic waves arrive, thereby reducing casualties and economic losses. The G-TEAM model is a data-driven national earthquake early warning system that integrates Graph Neural Networks (GNN) and Transformer architectures. It can rapidly estimate epicenter location, magnitude, and seismic intensity distribution within 3 seconds after earthquake occurrence. By directly processing raw seismic waveform data, the model eliminates limitations from manual feature selection and enhances prediction accuracy and real-time performance through multi-station data utilization.

This model is an efficient earthquake early warning system combining Graph Neural Networks (GNN) and Transformer architectures, taking seismic waveform data from any number of seismic stations as input. It enables real-time processing of seismic signals to deliver fast and precise estimations of hypocenter location, magnitude, and seismic intensity distribution range (characterized by Peak Ground Acceleration, PGA). Leveraging deep learning methods, the model fully exploits spatial correlations and temporal features within seismic networks to improve warning accuracy and response speed, providing robust support for earthquake emergency response and disaster mitigation strategies.

![](./images/image.png)

The PGA prediction architecture using multi-source seismic station data operates as follows:

1. The system receives position data and waveform recordings from multiple seismic stations, along with target coordinates for PGA estimation.  
2. For each station's waveform data:  
   - Perform standardization  
   - Extract features via Convolutional Neural Networks (CNN)  
   - Fuse features through fully connected layers  
   - Combine with station coordinates to form feature vectors  
3. Target PGA coordinates are processed through positional encoding to generate feature vectors.  
4. All feature vectors are sequentially fed into a Transformer encoder that captures global dependencies via self-attention mechanisms.  
5. Encoder outputs pass through three independent fully connected layers to perform regression tasks: magnitude estimation, epicenter localization, and PGA prediction.

## Training Data

The model is trained using the [Diting Dataset 2.0 - Multifunctional Large AI Training Dataset for China Seismic Network](http://www.esdc.ac.cn/article/137), which contains:

- Waveform records from 1,177 fixed stations in China (15°-50°N, 65°-140°E)  
- Data coverage: March 2020 to February 2023  
- 264,298 local seismic events (M > 0)  
- Only retains initial P-wave and S-wave phases  
- Includes events recorded by ≥3 stations for reliability  

The inference module has been open-sourced and supports prediction using provided checkpoint files (.ckpt).

## Quick Start

You can download the required data and ckpt files for training and inference at [dataset](https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/G-TEAM/)

### Execution

Run via command line using the `main` script:

```python
python main.py --cfg_path ./config/config.yaml --device_id 0 --device_target Ascend

```

Parameters:
--cfg_path: Configuration file path (default: "./config/config.yaml")
--device_target: Hardware type (default: Ascend)
--device_id: Device ID (default: 0)

### Visualization

![](./images/pga.png)

Scatter plot compares predicted vs actual PGA values (x-axis vs y-axis). Closer alignment to y=x line indicates higher accuracy.

### 结果展示

|   Parameter         |        NPU              |
|:----------------------:|:--------------------------:|
|   Hardware       |  Ascend, memory 64G    |
|   MindSpore Version       |  mindspore2.5.0    |
|   Dataset       |  diting2_2020-2022_sc    |
|   Test Parameters     |  batch_size=1<br>steps=9 |
| Magnitude Error (RMSE, MSE)     |   [ 0.262, 0.257 ]       |
| Epicenter Distance Error (RMSE, MAE)    |   [ 4.318 , 4.123 ]    |
| Hypocenter Depth Error (RMSE, MAE)    |   [ 5.559 , 5.171 ]    |
| PGA Error (RMSE, MSE) |[ 0.466, 0.287 ]  |
| Inference Resource       |        1NPU                    |
| Inference Speed(ms/step)  |     556                 |

## Contributors

gitee id: chengjie, longjundong, xujiabao, dinghongyang, funfunplus

email: funniless@163.com