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

- `2022.09.02` 中国商飞首席科学家吴光辉院士在WAIC2022世界人工智能大会发布首个工业级流体仿真大模型“东方.御风”, AI流体仿真助力国产大飞机气动仿真， [相关新闻](http://www.news.cn/fortune/2022-09/06/c_1128978806.htm)。

## **即将到来** 🚀

- 不要着急，精彩即将到来~

**更多应用案例请见**：👀

- [PDENet](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_plus_data_driven/variant_linear_coe_pde_net)
- [圆柱绕流](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physical_driven/flow_past_cylinder)
- [`N-S`方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes)
- [`Burgers`方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physical_driven/burgers_pinns)

## **安装教程**

### 版本依赖关系

由于MindFlow与MindSpore有依赖关系，请根据下表中所指示的对应关系，在[MindSpore下载页面](https://www.mindspore.cn/versions)下载并安装对应的whl包。

|    MindFlow    |                                        分支                                        |    MindSpore    | Python |
|:--------------:|:----------------------------------------------------------------------------------:|:---------------:|:------:|
|  0.1.0-alpha   | [r0.2.0-alpha](https://gitee.com/mindspore/mindscience/tree/r0.2.0-alpha/MindFlow) | \>=2.0.0-alpha  | \>=3.7 |

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

### pip安装(暂不可用)

```bash
pip install mindflow_[gpu|ascend]
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

## **贡献指南**

- 如何贡献您的代码，请点击此处查看：[贡献指南](https://gitee.com/mindspore/mindscience/blob/master/CONTRIBUTION.md)

## **许可证**

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
