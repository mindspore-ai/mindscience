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

- 🔥`2022.09.02` Academician Guanghui Wu, Chief Scientist of COMAC, released the first industrial flow simulation model "DongFang.YuFeng" at WAIC2022 World Artificial Intelligence Conference. AI flow simulation assisted the aerodynamic simulation of domestic large aircraft. [Page](http://www.news.cn/fortune/2022-09/06/c_1128978806.htm)

## **Coming Soon** 🚀

- Everything is coming soon, don't worry~

**More Cases**：👀

### CFD

- [Couetee flow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/couette)
- [1d lax tube](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/lax)
- [2d riemann](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/riemann2d)
- [Sod tube](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/sod)

### Data Driven

- [`N-S` equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes)
- [`Burgers` equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers)

### Physics Driven

- [Flow around a cylinder](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/flow_past_cylinder)
- [`Burgers-pinns` equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/burgers_pinns)
- [2D Darcy flow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/2D_Darcy)
- [poisson ring](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson_ring)
- [poisson pinns](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/possion_pinns)
- [simple pde introduction](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/sympy_pde_introduction)

### Physics Plus Data Driven

- [PDENet](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_plus_data_driven/variant_linear_coe_pde_net)

## **Installation**

### Version Dependency

Because MindFlow is dependent on MindSpore, please click [MindSpot Download Page](https://www.mindspore.cn/versions) according to the corresponding relationship indicated in the following table. Download and install the corresponding whl package.

| MindFlow |                                  Branch                                |    MindSpore   | Python |
|:--------:|:----------------------------------------------------------------------:|:--------------:|:------:|
|  0.1.0   | [master](https://gitee.com/mindspore/mindscience/tree/master/MindFlow) | \>=2.0.0-alpha | \>=3.7 |

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

### **pip install** (not support temporarily)

```bash
pip install mindflow_[gpu|ascend]
```

### **source code install**

```bash
git clone https://gitee.com/mindspore/mindscience.git
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

## **Contribution Guide**

- Please click here to see how to contribute your code:[Contribution Guide](https://gitee.com/mindspore/mindscience/blob/master/CONTRIBUTION.md)

## **License**

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
