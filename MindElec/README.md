# MindElec

[查看中文](README_CN.md)

<!-- TOC -->

- [MindElec](#mindelec)
    - [Introduction to MindElec](#introduction-to-mindelec)
        - [Data Building and Conversion](#data-building-and-conversion)
        - [Simulation](#simulation)
            - [Electromagnetic Model Library](#electromagnetic-model-library)
            - [Optimization strategy](#optimization-strategy)
        - [Result Visualization](#result-visualization)
        - [Publications](#publications)
    - [Installation Guide](#installation-guide)
        - [Confirming the System Environment Information](#confirming-the-system-environment-information)
        - [Installing Using pip](#installing-using-pip)
        - [Installing Using Source Code](#installing-using-source-code)
    - [API](#api)
    - [Installation Verification](#installation-verification)
    - [Quick Start](#quick-start)
    - [Documents](#documents)
    - [Community](#community)
        - [Governance](#governance)
    - [Contribution](#contribution)
    - [Release Notes](#release-notes)
    - [License](#license)

<!-- /TOC -->

## **Up-to-date News** 📰

- `2022.07` Our paper "A Universal PINNs Method for Solving Partial Differential Equations with a Point Source" was accepted by IJCAI 2022，please refer our [paper](https://www.ijcai.org/proceedings/2022/533) and [code](https://gitee.com/mindspore/mindscience/tree/master/MindElec/examples/physics_driven/time_domain_maxwell)

## Introduction to MindElec

Electromagnetic simulation refers to simulating the propagation characteristics of electromagnetic waves in objects or space through computation. It is widely used in scenarios such as mobile phone tolerance simulation, antenna optimization, and chip design. Conventional numerical methods, such as finite difference and finite element, require mesh segmentation and iterative computation. The simulation process is complex and the computation time is long, which cannot meet the product design requirements. With the universal approximation theorem and efficient inference capability, the AI method can improve the simulation efficiency.

MindElec is an AI electromagnetic simulation toolkit developed based on MindSpore. It consists of the electromagnetic model library, data build and conversion, simulation computation, and result visualization. End-to-end AI electromagnetic simulation is supported. Currently, Huawei has achieved phase achievements in the tolerance scenario of Huawei mobile phones. Compared with the commercial simulation software, the S parameter error of AI electromagnetic simulation is about 2%, and the end-to-end simulation speed is improved by more than 10 times.

<div align=center>
<img src="docs/MindElec-architecture-en.jpg" alt="MindElec Architecture" width="600"/>
</div>

### Data Building and Conversion

Supports geometric construction in constructive solid geometry (CSG) mode, such as the intersection set, union set, and difference set of rectangles and circles, and also supports efficient tensor conversion of CST and STP data (data formats supported by commercial software such as CST). In the future, we will support smart grid division for traditional scientific computing.

### Simulation

#### Electromagnetic Model Library

Provides the physical-driven and data-driven AI electromagnetic models. Physical-driven model refers to network training that does not require additional label data. Only equations and initial boundary conditions are required. Data-driven model refers to training that requires data generated through simulation or experiments. Compared with the data-driven model, the physical-driven model has the advantage of avoiding problems such as cost and mesh independence caused by data generation. The disadvantage of the physical-driven model is that the expression form of the equation needs to be specified and technical challenges such as point source singularity, multi-task loss function, and generalization need to be overcome.

#### Optimization strategy

Provides a series of optimization strategies to improve physical-driven and data-driven model accuracy and reduce training costs. Data compression can effectively reduce the storage and computation workload of the neural network. Multi-scale filtering and dynamic adaptive weighting can improve the model accuracy and overcome the problems such as point source singularity. Few-shot learning will be completed subsequently to reduce the training data volume and training cost.

### Result Visualization

The simulation results, such as the S parameters or electromagnetic fields, can be saved in the CSV or VTK files. MindInsight can display the loss function changes during the training process and display the results on the web page in the form of images. ParaView is the third-party open-source software and can dynamically display advanced functions such as slicing and flipping.

### Publications

If you are interested in solving time-domain Maxwell's equations, please read our [paper](https://arxiv.org/abs/2111.01394): Xiang Huang, Hongsheng Liu, Beiji Shi, Zidong Wang, Kang Yang, Yang Li, Bingya Weng, Min Wang, Haotian Chu, Jing Zhou, Fan Yu, Bei Hua, Lei Chen, Bin Dong, Solving Partial Differential Equations with Point Source Based on Physics-Informed Neural Networks, preprint 2021

If you are interested in our Meta-Auto-Decoder for solving parametric PDEs, please read our [paper](https://arxiv.org/abs/2111.08823): Xiang Huang, Zhanhong Ye, Hongsheng Liu, Beiji Shi, Zidong Wang, Kang Yang, Yang Li, Bingya Weng, Min Wang, Haotian Chu, Jing Zhou, Fan Yu, Bei Hua, Lei Chen, Bin Dong, Meta-Auto-Decoder for Solving Parametric Partial Differential Equations, preprint 2021

## Installation Guide

### Confirming the System Environment Information

| Hardware| Operating System| Status|
| :------------ | :-------------- | :--- |
| Ascend 910    | Ubuntu-x86      | ✔️   |
|               | Ubuntu-aarch64  | ✔️   |
|               | EulerOS-aarch64 | ✔️   |
|               | CentOS-x86      | ✔️   |
|               | CentOS-aarch64  | ✔️   |

- Install MindSpore by referring to [MindSpore Installation Guide](https://www.mindspore.cn/install/en). The version must be 1.5.0 or later.
- For other dependencies, see [requirements.txt](https://gitee.com/mindspore/mindscience/blob/master/MindElec/requirements.txt).

### Installing Using pip

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/mindscience/{arch}/mindscience_mindelec_ascend-{version}-{python_version}-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependencies of the MindElec installation package are automatically downloaded during the .whl package installation. For details about dependencies, see [setup.py](https://gitee.com/mindspore/mindscience/blob/master/MindElec/setup.py). Pointcloud data generation depends on [pythonocc](https://github.com/tpaviot/pythonocc-core), please install the dependencies by yourself.
> - `{version}` denotes the version of MindElec. For example, when you are installing MindElec 0.1.0, `{version}` should be 0.1.0.
> - `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, `{arch}`should be x86_64. If the system is ARM architecture 64-bit, then it should be aarch64.
> - `{python_version}` specifies the python version of which MindElec is built. If you wish to use Python3.7.5, `{python_version}` should be cp37-cp37m. If Python3.9.0 is used, it should be cp39-cp39.

### Installing Using Source Code

1. Download the source code from the code repository.

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindscience.git
    ```

2. Build and install MindElec.

    ```bash
    cd ~/MindElec
    bash build.sh
    pip install output/mindscience_mindelec_ascend-{version}-cp37-cp37m-linux_{x86_64}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

## API

For details about MindElec APIs, see the [API](https://www.mindspore.cn/mindscience/docs/en/master/mindelec.html) page.

## Installation Verification

Run the following command. If the error message `No module named 'mindelec'` is not displayed, the installation is successful.

```bash
python -c 'import mindelec'
```

## Quick Start

For details about how to quickly use the AI electromagnetic simulation toolkit for training and inference, see [MindElec Guide](https://www.mindspore.cn/mindscience/docs/en/master/mindelec/intro_and_install.html).

## Documents

For more details about the installation guides, tutorials, and APIs, see [MindElec Documents](https://gitee.com/mindspore/docs/tree/master/docs/mindscience).

## Community

### Governance

[MindSpore Open Governance](https://gitee.com/mindspore/community/blob/master/governance.md)

## Contribution

Make your contribution. For more details, please refer to our [Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)

## Release Notes

[RELEASE](https://gitee.com/mindspore/mindscience/blob/master/MindElec/RELEASE.md)

## License

[Apache License 2.0](LICENSE)
