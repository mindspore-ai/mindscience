[English](./README.md) | 简体中文

## 目录

- [目录](#目录)
- [GAD-MAL 描述](#gad-mal-描述)
    - [主要特性](#主要特性)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
    - [快速开始](#快速开始-1)
    - [管道流程](#管道流程)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [项目文件说明](#项目文件说明)
- [更多信息](#更多信息)

## GAD-MAL 描述

GAD-MALL是一个基于主动学习（Active Learning）和3D卷积神经网络（3D CNN）的深度学习框架，旨在处理多目标高维优化问题。该框架通过生成模型、有限元方法（FEM）和3D打印技术相结合，提供了一种高效的数据驱动设计方法，尤其适用于具有复杂结构的材料的优化设计。该框架特别应用于实现高效的建筑材料设计优化特别是针对复杂的多目标优化问题，如异质材料在生物工程和材料科学领域中的应用。可以完成像骨移植支架的设计，通过优化支架的弹性模量和屈服强度，实现了具有生物相容性和高机械强度的异质性架构。

### 主要特性

1. **生成架构设计（GAD）**: GAD利用编码器-解码器网络（autoencoder）生成具有未知特性的架构集。自动编码器通过无监督学习将高维设计空间的探索转化为低维空间，并有效地表示高维数据，使设计过程更为高效。

2. **多目标主动学习环（MALL）**: MALL通过迭代地查询有限元方法（FEM）来评估生成的数据集，逐步优化架构性能。该方法通过主动学习循环，不断更新训练数据，逐步提高模型预测的准确性。

1. **3D打印与测试**: 通过激光粉末床熔融技术制造ML设计的架构材料，并实验验证其机械性能。

>论文：Peng, B., Wei, Y., Qin, Y. et al. Machine learning-enabled constrained multi-objective design of architected materials. Nat Commun 14, 6630 (2023). https://doi.org/10.1038/s41467-023-42415-y

## 数据集

该项目使用的主要数据集包括以下文件：

- 输入数据:
    - `3D_CAE_Train.npy`: 用于3D卷积自编码器的训练数据，存储为NumPy数组。
    - `Matrix12.npy` 和 `Matrix60.npy`: 这些文件包含不同构建的矩阵数据，用于架构生成和优化过程。
    - `E.csv`: 包含材料弹性模量的数据文件。
    - `yield.csv`: 包含材料屈服强度的数据文件。

这些数据集用于支持GAD-MALL框架中各个模型的训练和测试。

- 数据下载:
    - `Matrix12.npy`，`E.csv` 和 `yield.csv`: 存放于 `./src/data` 目录下

        ```txt
        ├── data
        │   ├── E.csv
        │   ├── yield.csv
        │   ├── Matrix12.npy
        │   └── README.txt
        ```

    - `3D_CAE_Train.npy` 可通过README.txt中的[链接](https://drive.google.com/file/d/1BfmD4bsPS2hG5zm7XGLHc8lpUN_WqhgV/view?usp=share_link)下载。
    - `Matrix60.npy` 可通过README.txt中的[链接](https://drive.google.com/file/d/1VRH4X_mACxM82HoaplwV0ThaDiN3iPXm/view?usp=share_link)下载。
- **预处理**: 在使用前，需要对数据进行归一化处理，并可能需要对数据进行裁剪或补零操作以适配模型的输入尺寸。

## 环境要求

该项目基于MindSpore深度学习框架。以下是主要的（**测试/开发**）环境依赖：

- **硬件** (GPU)
    - 显卡：NVIDIA GeForce RTX 4060 Laptop GPU
    - 驱动：CUDA 12.3
    - CUDA: 11.6
    - CUDNN: 8.4.1
- **操作系统**:
    - Windows WSL Ubuntu-20.04
- **Python 版本**:
    - Python 3.9
- 框架
    - [MindSpore](https://www.mindspore.cn/install/)
- **依赖库**:
    - mindspore==2.2.14
    - numpy==1.23.5
    - scipy==1.13.1
    - pandas==2.2.2
    - matplotlib==3.9.1
    - tqdm==4.66.5
    - 安装依赖库可以通过以下命令：

        ```python3.9 -u pip install -r requirement.txt```

- 欲了解更多信息，请查看以下资源:
  - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r2.2/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 快速开始

### 快速开始

通过官网安装好MindSpore后，就可以开始训练和验证如下:

- 在 GPU 上运行
  ```bash run_GAD_MALL.sh```

### 管道流程

通过官网安装好MindSpore和上面需要的数据集后，就可以开始在 GPU 上运行训练,生成架构设计-多目标主动学习循环 (GAD-MALL) 管道,请按照以下步骤操作：

  1. 训练 3D-CAE 模型作为生成模型GAD-MALL，请在终端中运行以下行
    python3.9 3D_CAE_ms.py
  2. 训练 3D-CNN 模型作为 GAD-MALL 的替代模型GAD-MALL，请在终端中运行以下行
    python3.9 3D_CNN_ms.py
  3. 使用 GAD-MALL 搜索具有特定弹性模量和高屈服强度的高性能建筑材料，请在终端中运行以下行
    python3.9 -u GAD_MALL_Active_learning.py
  4. 完成 GAD-MALL 流程后，您将获得具有特定预测弹性模量（E=2500 MPa、E=5000 MPa）和最高预测屈服强度的孔隙度矩阵

## 脚本说明

### 脚本和示例代码

文件结构如下：

```text

├── gad_mall
│   ├── data                          # 数据文件
│   │   ├── E.csv                     # 包含材料弹性模量的数据文件
│   │   ├── yield.csv                 # 包含材料屈服强度的数据文件
│   │   ├── Matrix12.npy              # 矩阵数据，用于架构生成和优化过程
│   │   ├── (Matrix60.npy)            # 矩阵数据，用于架构生成和优化过程
│   │   ├── (3D_CAE_Train.npy)        # 用于3D卷积自编码器的训练数据，存储为NumPy数组
│   │   └── README.txt                # 数据下载地址
│   ├── model                         # checkpoint文件
│   ├── results                       # 实验结果存放
│   ├── src                           # 源代码
│   │   ├── 3D_CAE_ms.py              # 3D卷积自编码器的实现
│   │   ├── 3D_CNN_ms.py              # 3D卷积神经网络模型的实现
│   │   └── GAD_MALL_Active_learning.py # GAD-MALL框架的实现
│   ├── README.md                     # 英文模型说明
│   ├── README_CN.md                  # 中文模型说明
│   ├── run_GAD_MALL.sh               # 训练启动脚本
|   └── requirements.txt              # Python环境依赖文件

```

### 项目文件说明

- `3D_CNN_ms.py`：实现了基于3D卷积神经网络的模型，适用于处理三维数据集，特别是在高维多目标优化问题中的应用。该模型通过体素化处理输入数据，并利用3D卷积层提取高层信息，最终进行材料性能的预测。
- `GAD_MALL_Active_learning_ms.py`：实现了GAD-MALL框架的主动学习策略，用于在数据标注成本高的场景中优化模型。该脚本结合生成模型和有限元方法，通过主动学习迭代搜索高性能架构。
- `3D_CAE_ms.py`：实现了3D卷积自编码器，用于无监督学习中的特征提取或数据降维。该自编码器是生成架构设计（GAD）过程中的关键组成部分。通过编码器-解码器网络对输入数据进行低维表示，并重建原始数据。
- `data/`: 数据文件夹，包含训练和测试数据集。
- `models/`: 存放训练好的模型和权重文件。
- `results/`: 存放模型推理和评估的结果。
- `requirements.txt`: Python环境依赖文件。

## 更多信息

有关更多信息请参阅原项目说明[GAD-MALL](https://github.com/Bop2000/GAD-MALL/tree/main)

