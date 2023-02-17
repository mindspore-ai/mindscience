[ENGLISH](README.md) | 简体中文

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mindspore.svg)](https://pypi.org/project/mindspore)
[![PyPI](https://badge.fury.io/py/mindspore.svg)](https://badge.fury.io/py/mindspore)
[![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat-square)](https://github.com/mindspore-ai/mindspore/blob/master/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://gitee.com/mindspore/mindscience/pulls)

# **MindFlow**

- [MindFlow介绍](#MindFlow介绍)
- [最新消息](#最新消息)
- [安装教程](#安装教程)
    - [依赖安装](#依赖安装)
    - [硬件支持情况](#硬件支持情况)
    - [pip安装](#pip安装)
    - [源码安装](#源码安装)
- [社区](#社区)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## **MindFlow介绍**

流体仿真是指通过数值计算对给定边界条件下的流体控制方程进行求解，从而实现流动的分析、预测和控制，其在航空航天、船舶制造以及能源电力等行业领域的工程设计中应用广泛。传统流体仿真的数值方法如有限体积、有限差分等，主要依赖商业软件实现，需要进行物理建模、网格划分、数值离散、迭代求解等步骤，仿真过程较为复杂，计算周期长。AI具备强大的学习拟合和天然的并行推理能力，可以有效地提升流体仿真效率。

MindFlow是基于[昇思MindSpore](https://www.mindspore.cn/)开发的流体仿真领域套件，支持航空航天、船舶制造以及能源电力等行业领域的AI流场模拟，旨在于为广大的工业界科研工程人员、高校老师及学生提供高效易用的AI计算流体仿真软件。

<div align=center><img src="docs/mindflow_archi_cn.png" alt="MindFlow Architecture" width="700"/></div>

## **最新消息** 📰

- 🔥`2023.02.05` [MindFlow 0.1.0-alpha](https://mindspore.cn/mindflow/docs/zh-CN/r0.1.0-alpha/index.html) 版本发布。
- 🔥`2023.01.17` 推出[MindFlow-CFD](https://zhuanlan.zhihu.com/p/599592997)基于MindSpore的端到端可微分求解器，[详见](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/mindflow/cfd)。
- 🔥`2022.12.27` MindSpore团队同西安交大陈刚老师合作发表[Temporal predictions of periodic flows using a mesh transformation and deep learning-based strategy](https://www.sciencedirect.com/science/article/pii/S1270963822007556)文章于航空领域Top期刊`Aerospace Science and Technology`，论文作者为邓志文、刘红升、时北极、王紫东、于璠、刘子扬(西交)、陈刚(通讯)。
- 🔥`2022.09.02` 中国商飞首席科学家吴光辉院士在WAIC2022世界人工智能大会发布首个工业级流体仿真大模型“东方.御风”, AI流体仿真助力国产大飞机气动仿真， [相关新闻](http://www.news.cn/fortune/2022-09/06/c_1128978806.htm)。

**更多应用案例请见**：👀

### 物理驱动

- [一维Burgers问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/burgers)
- [二维圆柱绕流](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/cylinder_flow)
- [不同几何体下的二维和三维Poisson问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson)
- [二维Darcy问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/darcy)

### Mindflow特性

- [基于MindFlow定义符号化偏微分方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/features/sympy_pde_introduction)
- [作用于圆环的二维Poisson问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/features/poisson_ring)

### 数据驱动

- [基于FNO求解一维Burgers](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers)
- [基于FNO求解二维Navier-Stokes](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes)

### CFD

- [一维Lax激波管](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/lax)
- [一维Sod激波管](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/sod)
- [二维库埃特流](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/couette)
- [二维黎曼问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/riemann2d)

### 数据机理融合

- [PDE-Net求解对流扩散方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/variant_linear_coe_pde_net)

## **安装教程**

### 版本依赖关系

由于MindFlow与MindSpore有依赖关系，请根据下表中所指示的对应关系，在[MindSpore下载页面](https://www.mindspore.cn/versions)下载并安装对应的whl包。

| MindFlow |                                   分支                                 |    MindSpore   | Python |
|:--------:|:----------------------------------------------------------------------:|:--------------:|:------:|
|  0.1.0   | [master](https://gitee.com/mindspore/mindscience/tree/master/MindFlow) | \>=2.0.0-alpha | \>=3.7 |

### 依赖安装

```bash
pip install -r requirements.txt
```

### 硬件支持情况

| 硬件平台      | 操作系统        | 状态 |
| :------------ | :-------------- | :--- |
| Ascend 910    | Ubuntu-x86      | ✔️ |
|               | Ubuntu-aarch64  | ✔️ |
|               | EulerOS-aarch64 | ✔️ |
|               | CentOS-x86      | ✔️ |
|               | CentOS-aarch64  | ✔️ |
| GPU CUDA 11.1 | Ubuntu-x86      | ✔️ |

### pip安装

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

### 源码安装

```bash
git clone https://gitee.com/mindspore/mindscience.git
cd {PATH}/mindscience/MindFlow
```

- 昇腾Ascend后端

```bash
bash build.sh -e ascend -j8
```

- GPU后端

```bash
export CUDA_PATH={your_cuda_path}
bash build.sh -e gpu -j8
```

- 安装编译所得whl包

```bash
cd {PATH}/mindscience/MindFLow/output
pip install mindflow_*.whl
```

## **社区**

### SIG 🏠

### 核心贡献者 🧑‍🤝‍🧑

感谢以下开发者做出的贡献：

yufan, wangzidong, liuhongsheng, zhouhongye, zhangyi, dengzhiwen, liulei, libokai, yangge, longzichao, yqiuu, haojiwei, leiyixiang

## **贡献指南**

- 如何贡献您的代码，请点击此处查看：[贡献指南](https://gitee.com/mindspore/mindscience/blob/master/CONTRIBUTION.md)

## **许可证**

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
