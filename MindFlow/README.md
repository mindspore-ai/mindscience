 ENGLISH | [简体中文](README_CN.md)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mindspore.svg)](https://pypi.org/project/mindspore)
[![PyPI](https://badge.fury.io/py/mindspore.svg)](https://badge.fury.io/py/mindspore)
[![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat-square)](https://github.com/mindspore-ai/mindspore/blob/master/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://gitee.com/mindspore/mindscience/pulls)

# **MindFlow**

- [Introduction](#Introduction)
- [Latest News](#Latest)
- [Installation](#Installation)
    - [Dependency](#Dependency)
    - [Hardware](#Hardware)
    - [pip install](#pip)
    - [source code install](#source)
- [Community](#Community)
- [Contribution Guide](#Contribution)
- [License](#License)

## **Introduction**

Flow simulation aims to solve the fluid governing equation under a given boundary condition by numerical methods, so as to realize the flow analysis, prediction and control. It is widely used in engineering design in aerospace, ship manufacturing, energy and power industries. The numerical methods of traditional flow simulation, such as finite volume method and finite difference method, are mainly implemented by commercial software, requiring physical modeling, mesh generation, numerical dispersion, iterative solution and other steps. The simulation process is complex and the calculation cycle is long. AI has powerful learning fitting and natural parallel inference capabilities, which can improve the efficiency of the flow simulation.

MindSpore Flow is a flow simulation suite developed based on [MindSpore](https://www.mindspore.cn/). It supports AI flow simulation in industries such as aerospace, ship manufacturing, and energy and power. It aims to provide efficient and easy-to-use AI computing flow simulation software for industrial research engineers, university professors, and students.

<div align=center><img src="docs/mindflow_archi_en.png" alt="MindFlow Architecture" width="700"/></div>

## **Latest News** 📰

- 🔥`2023.02.05` [MindFlow 0.1.0-alpha](https://mindspore.cn/mindflow/docs/zh-CN/r0.1.0-alpha/index.html) is released.
- 🔥`2023.01.17` [MindFlow-CFD](https://zhuanlan.zhihu.com/p/599592997), an End-to-End Differentiable Solver based on MindSpore，[see more](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow/mindflow/cfd).
- 🔥`2022.12.27` MindSpore team cooperates with Xi'an Jiaotong University teacher Gang Chen publish [Temporal predictions of periodic flows using a mesh transformation and deep learning-based strategy](https://www.sciencedirect.com/science/article/pii/S1270963822007556)in top journals `Aerospace Science and Technology`, authors are Zhiwen Deng, Hongsheng Liu, Beiji Shi, Zidonog Wang, Fan Yu, Ziyang Liu、Gang Chen(Corresponding author).
- 🔥`2022.09.02` Academician Guanghui Wu, Chief Scientist of COMAC, released the first industrial flow simulation model "DongFang.YuFeng" at WAIC2022 World Artificial Intelligence Conference. AI flow simulation assisted the aerodynamic simulation of domestic large aircraft. [Page](http://www.news.cn/fortune/2022-09/06/c_1128978806.htm).

**More Cases**：👀

### Physics Driven

- [1D Burgers](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow/applications/physics_driven/burgers_pinns)
- [2D Cylinder Flow](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow/applications/physics_driven/flow_past_cylinder)
- [2D Poisson on a Ring](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow/applications/physics_driven/poisson_ring)
- [Definition of Symbolic PDE Based on MindFlow](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow/applications/physics_driven/sympy_pde_introduction)

### Data Driven

- [FNO for 1D Burgers](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow/applications/data_driven/burgers)
- [FNO for 2D Navier-Stokes](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow/applications/data_driven/navier_stokes)

### CFD

- [1D Lax Tube](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow/applications/cfd/lax)
- [1D Sod Tube](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow/applications/cfd/sod)
- [2D Couette Flow](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow/applications/cfd/couette)
- [2D Riemann](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow/applications/cfd/riemann2d)

### Physics Plus Data Driven

- [PDE-Net for Convecton-Diffusion Equation](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow/applications/physics_plus_data_driven/variant_linear_coe_pde_net)

## **Installation**

### Version Dependency

Because MindFlow is dependent on MindSpore, please click [MindSpot Download Page](https://www.mindspore.cn/versions) according to the corresponding relationship indicated in the following table. Download and install the corresponding whl package.

|    MindFlow    |                                       Branch                                       |    MindSpore    | Python |
|:--------------:|:----------------------------------------------------------------------------------:|:---------------:|:------:|
|  0.1.0-alpha   | [r0.2.0-alpha](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow) | \>=2.0.0-alpha  | \>=3.7 |

### Install Dependency

```bash
pip install -r requirements.txt
```

### Hardware

| Hardware      | OS              | Status |
|:--------------| :-------------- | :--- |
| Ascend 910    | Ubuntu-x86      | ✔️ |
|               | Ubuntu-aarch64  | ✔️ |
|               | EulerOS-aarch64 | ✔️ |
|               | CentOS-x86      | ✔️ |
|               | CentOS-aarch64  | ✔️ |
| GPU CUDA 11.1 | Ubuntu-x86      | ✔️ |

### **pip install**

```bash
export MS_VERSION=2.0.0a0
export MindFlow_VERSION=0.1.0a0
# gpu and ascend are supported
export DEVICE_NAME=gpu
# cuda-10.1 and cuda-11.1 are supported
export CUDA_VERSION=cuda-11.1

# Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindScience/${DEVICE_NAME}/x86_64/${CUDA_VERSION}/mindflow_${DEVICE_NAME}-${MindFlow_VERSION}-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindScience/${DEVICE_NAME}/x86_64/${CUDA_VERSION}/mindflow_${DEVICE_NAME}-${MindFlow_VERSION}-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindScience/${DEVICE_NAME}/x86_64/${CUDA_VERSION}/mindflow_${DEVICE_NAME}-${MindFlow_VERSION}-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### **source code install**

```bash
git clone https://gitee.com/mindspore/mindscience.git -b r0.2.0-alpha
cd {PATH}/mindscience/MindFlow
```

- Ascend backend

```bash
bash build.sh -e ascend -j8
```

- GPU backend

```bash
export CUDA_PATH={your_cuda_path}
bash build.sh -e gpu -j8
```

- Install whl package

```bash
cd {PATH}/mindscience/MindFLow/output
pip install mindflow_*.whl
```

## **Community**

### CO-CHAIR

### Core Contributor 🧑‍🤝‍🧑

Thanks goes to these wonderful people:

yufan, wangzidong, liuhongsheng, zhouhongye, zhangyi, dengzhiwen, liulei, libokai, yangge, longzichao, yqiuu, haojiwei, leiyixiang

## **Contribution Guide**

- Please click here to see how to contribute your code:[Contribution Guide](https://gitee.com/mindspore/mindscience/blob/master/CONTRIBUTION.md)

## **License**

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
