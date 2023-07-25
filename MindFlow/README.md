 ENGLISH | [简体中文](README_CN.md)

[![master](https://img.shields.io/badge/version-master-blue.svg?style=flat?logo=Gitee)](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/README.md)
[![docs](https://img.shields.io/badge/docs-master-yellow.svg?style=flat)](https://mindspore.cn/mindflow/docs/en/master/index.html)
[![internship](https://img.shields.io/badge/internship-tasks-important.svg?style=flat)](https://gitee.com/mindspore/community/issues/I55B5A?from=project-issue)
[![SIG](https://img.shields.io/badge/community-SIG-yellowgreen.svg?style=flat)](https://mindspore.cn/community/SIG/detail/en?name=mindflow%20SIG)
[![Downloads](https://static.pepy.tech/badge/mindflow-gpu)](https://pepy.tech/project/mindflow-gpu)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://gitee.com/mindspore/mindscience/pulls)
[![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat)](https://github.com/mindspore-ai/mindspore/blob/master/LICENSE)

# **MindFlow**

- [Introduction](#Introduction)
- [Latest News](#Latest-News)
- [MindFlow Features](#MindFlow-Features)
- [Applications](#Applications)
    - [Physics Driven](#Physics-Driven)
    - [Data Driven](#Data-Driven)
    - [Data-Mechanism Fusion](#Data-Mechanism-Fusion)
    - [CFD](#CFD)
- [Installation](#Installation)
    - [Version Dependency](#Version-Dependency)
    - [Install Dependency](#Install-Dependency)
    - [Hardware](#Hardware)
    - [pip install](#pip-install)
    - [source code install](#source-code-install)
- [Community](#Community)
    - [SIG](#Join-MindFlow-SIG)
    - [Core Contributor](#Core-Contributor)
    - [Community Partners](#Community-Partners)
- [Contribution Guide](#Contribution-Guide)
- [License](#License)

## **Introduction**

Flow simulation aims to solve the fluid governing equation under a given boundary condition by numerical methods, so as to realize the flow analysis, prediction and control. It is widely used in engineering design in aerospace, ship manufacturing, energy and power industries. The numerical methods of traditional flow simulation, such as finite volume method and finite difference method, are mainly implemented by commercial software, requiring physical modeling, mesh generation, numerical dispersion, iterative solution and other steps. The simulation process is complex and the calculation cycle is long. AI has powerful learning fitting and natural parallel inference capabilities, which can improve the efficiency of the flow simulation.

MindSpore Flow is a flow simulation suite developed based on [MindSpore](https://www.mindspore.cn/). It supports AI flow simulation in industries such as aerospace, ship manufacturing, and energy and power. It aims to provide efficient and easy-to-use AI computing flow simulation software for industrial research engineers, university professors, and students.

<div align=center><img src="docs/mindflow_archi_en.png" alt="MindFlow Architecture" width="700"/></div>

## **Latest News**

- 🔥`2023.07.21` Mindspore team has cooperated with the COMAC team and Professor Dong Bin from Peking University in the respect of proposing a [Prediction of transactional flow over supercritical airfoils using geometric-encoding and deep-learning strategies](https://doi.org/10.1063/5.0155383).The article is published in the top journal Physics of Fluids in the field of fluids, which elaborates on the technical solution of "Dongfang Yufeng". The authors of the paper are Deng Zhiwen, Wang Jing (Shanghai Jiao Tong University, COMAC), Liu Hongsheng, Xie Hairun (COMAC), Li Bokai, Zhang Miao (COMAC), Jia Tingmeng, Zhang Yi, Wang Zidong, and Dong Bin (Peking University, Corresponding author). For example code, please refer to: [DongFang. YuFeng](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_steady)
- 🔥`2023.07.18` Mindspore team, in collaboration with Professor Sun Hao of the Hillhouse Institute of Artificial Intelligence of Renmin University of China, based on the basic software and hardware platform of Shengteng AI and the MindSpore AI framework, has proposed a physics encoded recursive Convolutional neural network (PeRCNN). This achievement has been published in Nature Machine Intelligence, a sub journal of Nature. [Encoding physics to learn reaction-diffusion processes](https://www.nature.com/articles/s42256-023-00685-7).The relevant code has been opened in the MindSpore Flow code repository of the open source community Gitee. Here is a sample code:[PeRCNN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/PeRCNN)
- 🔥`2023.07.06` The 2023 World Artificial Intelligence Conference with the theme of "Connect the World Intelligently. Generate the Future" was successfully held at the Shanghai World Expo Center. The 3D Supercritical airfoil fluid simulation AI model "Dongfang Yifeng" from Comac Shanghai Aircraft Design and Research Institute won the SAIL Award, the highest award of the World Artificial Intelligence Conference. This model is a large intelligent AI model for wing complex flow simulation scenarios jointly developed by Comac Co., Ltd. Shanghai Aircraft Design and Research Institute and Huawei based on the domestic Shengteng AI basic software and hardware platform and MindSpore AI framework, [Page](https://www.thepaper.cn/newsDetail_forward_23769936).
- 🔥`2023.05.21` The second plenary meeting of the intelligent fluid mechanics industrial consortium was successfully held in Hangzhou West Lake University, and Shengsi MindSpore co organized the meeting. Three academicians of the CAS Member, representatives of the industrial consortium and experts from the academic and industrial circless who care about the consortium attended the meeting. The first fluid mechanics model for aircraft - "Qinling · AoXiang" model is pre released. This model is an intelligent model for aircraft fluid simulation jointly developed by the International Joint Institute of fluid mechanics Intelligence of Northwestern Polytechnical University and Huawei based on the domestic Shengteng AI basic software and hardware platform and MindSpore AI framework.[Page](http://science.china.com.cn/2023-05/23/content_42378458.htm)。
- 🔥`2023.02.28` Mindspore team has cooperated with Prof. Bin Dong from Peking University and Prof. Yanli Wang from CSRC in the respect of proposing a neural sparse representation to solve Boltzmann equation. Our achievement is about to publish. [Solving Boltzmann equation with neural sparse representation](https://arxiv.org/abs/2302.09233). Here is a sample code:[Neural representation method for Boltzmann equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann)
- 🔥`2023.02.05` [MindFlow 0.1.0-alpha](https://mindspore.cn/mindflow/docs/zh-CN/r0.1.0-alpha/index.html) is released.
- 🔥`2023.01.17` [MindFlow-CFD](https://zhuanlan.zhihu.com/p/599592997), an End-to-End Differentiable Solver based on MindSpore，[see more](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/mindflow/cfd).
- 🔥`2022.12.27` MindSpore team cooperates with Xi'an Jiaotong University teacher Gang Chen publish [Temporal predictions of periodic flows using a mesh transformation and deep learning-based strategy](https://www.sciencedirect.com/science/article/pii/S1270963822007556) in top journals `Aerospace Science and Technology`, authors are Zhiwen Deng, Hongsheng Liu, Beiji Shi, Zidonog Wang, Fan Yu, Ziyang Liu、Gang Chen(Corresponding author).
- 🔥`2022.09.02` Academician Guanghui Wu, Chief Scientist of COMAC, released the first industrial flow simulation model "DongFang.YuFeng" at WAIC2022 World Artificial Intelligence Conference. AI flow simulation assisted the aerodynamic simulation of domestic large aircraft. [Page](http://www.news.cn/fortune/2022-09/06/c_1128978806.htm).

## MindFlow Features

- [MindSpore Grad](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/features/mindspore_grad_cookbook.ipynb)

- [Solve Pinns by MindFlow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/features/solve_pinns_by_mindflow)

## Applications

### Physics Driven

|        Case            |        Dataset               |    Network       |  GPU    |  NPU  |
|:----------------------:|:--------------------------:|:---------------:|:-------:|:------:|
|[Burgers Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/burgers)       |         [Burgers Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/)              |     PINNs        |   ✔️     |   ✔️   |
|[2D Cylinder Flow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/cylinder_flow)        |     [2D Cylinder Fow Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/flow_past_cylinder/)              |        PINNs     |     ✔️   |   ✔️   |
|[2D Darcy](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/darcy)      |             -              |      PINNs      |  ✔️      |  ✔️    |
|[Poisson Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson)       |             -              |        PINNs     |  ✔️      |   ✔️   |
|[Boltzmann Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann)      |             -              |      PINNs       |   ✔️     |   ✔️   |
|[2D Taylor-Green Votex](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/taylor_green/2d)  |             -              |      PINNs        |   ✔️     |   ✔️   |
|[Navier-Stoken Inverse](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes_inverse)  |            [Navier-Stoken Inverse Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/inverse_navier_stokes/)             |       PINNs       |   ✔️     |   ✔️   |
|[2D Poisson Equation with Point Source](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson_point_source)      |             -              |       PINNs       |   ✔️     |   ✔️   |

### Data Driven

|        Case            |              Dataset                  |    Network       |  GPU    |  NPU  |
|:----------------------:|:-------------------------------------:|:---------------:|:-------:|:------:|
|[DongFang.YuFeng](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_steady)   |  [2D Airfoil Flow Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/airfoil/2D_steady/)   |     ViT           |   ✔️     |   ✔️   |
|[FNO Solve Burgers Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers_fno)   | [1D Burgers Equation Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/)     |     FNO1D       |   ✔️     |   ✔️   |
|[KNO Solve Burgers Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers_kno)     | [1D Burgers Equation Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/)       |       KNO1D       |   ✔️     |   ✔️   |
|[FNO Solve Navier-Stokes Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes_fno)   |  [2D Navier-Stokes Equation Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)         |        FNO2D          | ✔️   |   ✔️    |
|[FNO3d Solve Navier-Stokes Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes_3d_fno)  | [2D Navier-Stokes Equation Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)          |          FNO3D        |   ✔️     |   ✔️   |
|[KNO Solve Navier-Stokes Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes_kno)    |  [2D Navier-Stokes Equation Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)         |        KNO2D          |   ✔️     |   ✔️   |
|[2D Riemann Problem](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/cae_lstm)       |  [2D Riemann Problem Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/riemann/)       |     CAE-LSTM      |   ✔️     |   ✔️   |
|[shu-osher Problem](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/cae_lstm)     |   [1D shu-osher Problem Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/shu_osher/)    |      CAE-LSTM      |   ✔️     |   ✔️   |
|[sod Problem](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/cae_lstm)   |  [1D sod Problem Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/sod/)         |     CAE-LSTM    |   ✔️     |   ✔️   |
|[KH pelblem](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/cae_lstm/)         |  [2D K-H Problem Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/kh/)            |  CAE-LSTM     |   ✔️     |   ✔️   |
|[2D Airfoil Buffet](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/transonic_buffet_ehdnn)          |  [2D Airfoil Buffet Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/airfoil/2D_unsteady/)           |      ehdnn    |   ✔️     |   ✔️   |
|[Move Boundary Hdnn](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/move_boundary_hdnn)          |  [Move Boundary Hdnn Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/move_boundary_hdnn)           |      hdnn    |   ✔️     |   ✔️   |

### Data-Mechanism Fusion

|          Case              |        Dataset               |    Network       |  GPU    |  NPU  |
|:--------------------------:|:--------------------------:|:---------------:|:-------:|:------:|
|   [PDE-Net for Convection-Diffusion Equation](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/variant_linear_coe_pde_net)   | -   |    PDE-Net    |   ✔️     |   ✔️   |
|   [PeRCNN](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/PeRCNN)    |    [PeRCNN Dataset](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN/) | PeRCNN  |   ✔️     |   ✔️   |

### CFD

|       Case         |     格式      |    GPU    |    NPU |
|:-----------------:|:-------------:|:---------:|:-------|
|[sod shock tube](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/sod)      |    Rusanov    |       ✔️   |   -   |
|[lax shock tube](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/lax)      |    Rusanov    |      ✔️    |   -   |
|[2D Riemann Problem](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/riemann2d)  |       -       |     ✔️     |   -  |
|[Couette Flow](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/couette)        |       -       |  ✔️        |   -   |

## **Installation**

### Version Dependency

Because MindFlow is dependent on MindSpore, please click [MindSpore Download Page](https://www.mindspore.cn/versions) according to the corresponding relationship indicated in the following table. Download and install the corresponding whl package.

| MindFlow |                                  Branch                                |  MindSpore  |Python |
|:--------:|:----------------------------------------------------------------------:|:-----------:|:-------:|
|  master  | [master](https://gitee.com/mindspore/mindscience/tree/master/MindFlow) |        \       | \>=3.7 |
| 0.1.0rc1 | [r0.2.0](https://gitee.com/mindspore/mindscience/tree/r0.2.0/MindFlow) |   \>=2.0.0rc1  | \>=3.7 |

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

# GPU version
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/MindScience/gpu/x86_64/cuda-11.1/mindflow_gpu-0.1.0rc1-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Ascend version
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/MindScience/ascend/aarch64/mindflow_ascend-0.1.0rc1-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
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

### Join MindFlow SIG

<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html" charset="utf-8">
</head>
<body>

<table id="t2" style="text-align:center" align="center">
    <tr id="tr2">
        <td align="center">
            <img src="docs/co-chairs/张伟伟.jpeg" width="200" height="243"/>
            <p align="center">
                Northwestern Polytechnical University ZhangWeiwei
            </p>
        </td>
        <td align="center">
            <img src="docs/co-chairs/董彬.jpeg" width="200" height="243"/>
            <p align="center">
                Peking University DongBin
            </p>
        </td>
        <td align="center">
            <img src="docs/co-chairs/孙浩.jpeg" width="200" height="243"/>
            <p align="center">
                RenMin University of China SunHao
            </p>
        </td>
    </tr>
</table>
</body>
</html>

[Join](https://mp.weixin.qq.com/s/e00lvKx30TsqjRhYa8nlhQ) MindSpore [MindFlow SIG](https://mindspore.cn/community/SIG/detail/?name=mindflow%20SIG) to help AI fluid simulation development.
MindSpore AI for Science, [Learning and Learning to solve PDEs](https://www.bilibili.com/video/BV1ur4y1H7vB?p=4) topic report by Dong Bin, Peking University.
We will continue to release [open source internship tasks](https://gitee.com/mindspore/community/issues/I55B5A?from=project-issue), build MindFlow ecology with you, and promote the development of computational fluid dynamics with experts, professors and students in the field. Welcome to actively claim the task.

### Core Contributor

Thanks goes to these wonderful people 🧑‍🤝‍🧑:

yufan, wangzidong, liuhongsheng, zhouhongye, zhangyi, dengzhiwen, liulei, guoboqiang, chengzeruizhi, libokai, yangge, longzichao, qiuyisheng, haojiwei, leiyixiang

### Community Partners

<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
</head>
<body>
<table id="t1" style="text-align:center" align="center">
    <tr id="tr1">
        <td>
            <img src="docs/partners/CACC.jpeg"/>
            <p align="center">
                Commercial Aircraft Corporation of China Ltd
            </p>
        </td>
        <td>
            <img src="docs/partners/TaiHuLaboratory.jpeg"/>
            <p align="center">
                TaiHu Laboratory
            </p>
        </td>
        <td>
            <img src="docs/partners/NorthwesternPolytechnical.jpeg"/>
            <p align="center">
                Northwestern Polytechnical University
            </p>
        </td>
        <td>
            <img src="docs/partners/Peking_University.jpeg"/>
            <p align="center">
                Peking University
            </p>
        </td>
        <td>
            <img src="docs/partners/RenminUniversity.jpeg"/>
            <p align="center">
                Renmin University of China
            </p>
        </td>
    </tr>
</table>
</body>
</html>

## **Contribution Guide**

- Please click here to see how to contribute your code:[Contribution Guide](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/CONTRIBUTION_CN.md)

## **License**

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
