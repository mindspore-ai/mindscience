# MindElec

[View English](README.md)

<!-- TOC -->

- [MindElec](#mindelec)
    - [Introduction to MindElec](#introduction-to-mindelec)
        - [Data Build and Conversion](#data-build-and-conversion)
        - [Simulation](#simulation)
            - [Electromagnetic Model Library](#electromagnetic-model-library)
            - [Optimization strategy](#optimization-strategy)
        - [Result Visualization](#result-visualization)
    - [Installation Guide](#installation-guide)
        - [Confirming the System Environment Information](#confirming-the-system-environment-information)
        - [Installing Using pip](#installing-using-pip)
            - [Installing MindSpore](#installing-mindspore)
            - [Installing MindElec](#installing-mindelec)
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

## Introduction to MindElec

Electromagnetic simulation refers to simulating the propagation characteristics of electromagnetic waves in objects or space through computation. It is widely used in scenarios such as mobile phone tolerance simulation, antenna optimization, and chip design. Conventional numerical methods, such as finite difference and finite element, require mesh segmentation and iterative computation. The simulation process is complex and the computation time is long, which cannot meet the product design requirements. With the universal approximation theorem and efficient inference capability, the AI method can improve the simulation efficiency.

MindElec is an AI electromagnetic simulation toolkit developed based on MindSpore. It consists of the electromagnetic model library, data build and conversion, simulation computation, and result visualization. End-to-end AI electromagnetic simulation is supported. Currently, Huawei has achieved phase achievements in the tolerance scenario of Huawei mobile phones. Compared with the commercial simulation software, the S parameter error of AI electromagnetic simulation is about 2%, and the end-to-end simulation speed is improved by more than 10 times.

<img src="docs/MindElec-architecture.jpg" alt="MindElec Architecture" width="600"/>

### Data Build and Conversion

Supports geometric construction in constructive solid geometry (CSG) mode, such as the intersection set, union set, and difference set of rectangles and circles, and efficient tensor conversion of CST and STP data (data formats supported by commercial software such as CST).

### Simulation

#### Electromagnetic Model Library

Includes the physical-driven and data-driven AI electromagnetic models. Physical-driven model refers to network training that does not require additional label data. Only equations and initial boundary conditions are required. Data-driven model refers to training that requires data generated through simulation or experiments. Compared with the data-driven model, the physical-driven model has the advantage of avoiding problems such as cost and mesh independence caused by data generation. The disadvantage of the physical-driven model is that the expression form of the equation needs to be specified and technical challenges such as point source singularity, multi-task loss function, and generalization need to be overcome.

#### Optimization strategy

Receives tensor data of AI affinity and performs physical-driven and data-driven electromagnetic simulations. Provides a series of optimization measures to improve model accuracy and reduce training costs. Data compression can effectively reduce the storage and computation workload of the neural network. Multi-scale filtering and dynamic adaptive weighting can improve the model accuracy and overcome the problems such as point source singularity. Few-shot learning will be completed subsequently to reduce the training data volume and training cost.

### Result Visualization

The simulation results, such as the S parameters or electromagnetic fields, can be saved in the CSV or VTK files. MindInsight can display the loss function changes during the training process and display the results on the web page in the form of images. ParaView is the third-party open-source software and can dynamically display advanced functions such as slicing and flipping.

## Installation Guide

### Confirming the System Environment Information

| Hardware| Operating System| Status|
| :------------ | :-------------- | :--- |
| Ascend 910    | Ubuntu-x86      | ✔️   |
|               | Ubuntu-aarch64  | ✔️   |
|               | EulerOS-aarch64 | ✔️   |
|               | CentOS-x86      | ✔️   |
|               | CentOS-aarch64  | ✔️   |

- Install MindSpore by referring to [MindSpore Installation Guide](https://www.mindspore.cn/install/en). The version must be 1.4.0 or later.
- For other dependencies, see [setup.py](https://gitee.com/mindspore/mindscience/blob/master/MindElec/setup.py).

### Installing Using pip

#### Installing MindSpore

```bash
pip install https://hiq.huaweicloud.com/download/mindspore/ascend/x86_64/mindspore-1.4.0-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - Select a proper installation package based on the Python version on the local host. For example, if the Python version on the local host is 3.7, change `cp38-cp38` in the preceding command to `cp37-cp37m`.

#### Installing MindElec

```bash
pip install https://hiq.huaweicloud.com/download/mindscience/x86_64/mindscience_mindelec_ascend-0.1.0-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - When the network is connected, dependencies of the MindElec installation package are automatically downloaded during the .whl package installation. For details about dependencies, see [setup.py](https://gitee.com/mindspore/mindscience/blob/master/MindElec/setup.py). In other cases, install the dependencies by yourself.

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

For details about MindElec APIs, see the [API](https://www.mindspore.cn/mindelec/api/en/master/index.html) page.

## Installation Verification

Run the following command. If the error message `No module named 'mindelec'` is not displayed, the installation is successful.

```bash
python -c 'import mindelec'
```

## Quick Start

For details about how to quickly use the AI electromagnetic simulation toolkit for training and inference, see [MindElec Guide](https://www.mindspore.cn/mindelec/docs/en/master/index.html).

## Documents

For more details about the installation guides, tutorials, and APIs, see [MindSpore Documents](https://gitee.com/mindspore/docs).

## Community

### Governance

[MindSpore Open Governance](https://gitee.com/mindspore/community/blob/master/governance.md)

## Contribution

Make your contribution. For more details, please refer to our [Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)

## Release Notes

[RELEASE](https://gitee.com/mindspore/mindscience/blob/master/MindElec/RELEASE.md)

## License

[Apache License 2.0](LICENSE)
