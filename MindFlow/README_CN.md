[ENGLISH](README.md) | 简体中文

[![master](https://img.shields.io/badge/version-master-blue.svg?style=flat?logo=Gitee)](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/README_CN.md)
[![docs](https://img.shields.io/badge/docs-master-yellow.svg?style=flat)](https://mindspore.cn/mindflow/docs/zh-CN/master/index.html)
[![internship](https://img.shields.io/badge/internship-tasks-important.svg?style=flat)](https://gitee.com/mindspore/community/issues/I55B5A?from=project-issue)
[![SIG](https://img.shields.io/badge/community-SIG-yellowgreen.svg?style=flat)](https://mindspore.cn/community/SIG/detail/?name=mindflow%20SIG)
[![Downloads](https://static.pepy.tech/badge/mindflow-gpu)](https://pepy.tech/project/mindflow-gpu)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://gitee.com/mindspore/mindscience/pulls)
[![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat)](https://github.com/mindspore-ai/mindspore/blob/master/LICENSE)

# **MindFlow**

- [MindFlow介绍](#MindFlow介绍)
- [最新消息](#最新消息)
- [MindFlow特性](#Mindflow特性)
- [应用案例](#应用案例)
    - [物理驱动](#物理驱动)
    - [数据驱动](#数据驱动)
    - [数据机理融合](#数据机理融合)
    - [CFD](#CFD)
- [安装教程](#安装教程)
    - [依赖安装](#依赖安装)
    - [硬件支持情况](#硬件支持情况)
    - [pip安装](#pip安装)
    - [源码安装](#源码安装)
- [社区](#社区)
  - [SIG](#加入MindFlow-SIG)
  - [核心贡献者](#核心贡献者)
  - [合作伙伴](#合作伙伴)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## **MindFlow介绍**

流体仿真是指通过数值计算对给定边界条件下的流体控制方程进行求解，从而实现流动的分析、预测和控制，其在航空航天、船舶制造以及能源电力等行业领域的工程设计中应用广泛。传统流体仿真的数值方法如有限体积、有限差分等，主要依赖商业软件实现，需要进行物理建模、网格划分、数值离散、迭代求解等步骤，仿真过程较为复杂，计算周期长。AI具备强大的学习拟合和天然的并行推理能力，可以有效地提升流体仿真效率。

MindFlow是基于[昇思MindSpore](https://www.mindspore.cn/)开发的流体仿真领域套件，支持航空航天、船舶制造以及能源电力等行业领域的AI流场模拟，旨在于为广大的工业界科研工程人员、高校老师及学生提供高效易用的AI计算流体仿真软件。

<div align=center><img src="docs/mindflow_archi_cn.png" alt="MindFlow Architecture" width="700"/></div>

## **最新消息**

- 🔥`2023.05.21` 2023年5月21日,智能流体力学产业联合体第二次全体会议在杭州西湖大学成功举办，昇思MindSpore协办本次会议，三位中国科学院院士、产业联合体代表及关心联合体的学术界、产业界专家共计百位嘉宾现场参会。面向飞行器的首个流体力学大模型————“秦岭·翱翔”大模型预发布，该模型是由西北工业大学流体力学智能化国际联合研究所与华为基于国产昇腾AI基础软硬件平台及昇思MindSpore AI框架，共同研发的面向飞行器流体仿真的智能化模型，[相关新闻](http://science.china.com.cn/2023-05/23/content_42378458.htm)。
- 🔥`2023.02.28` Mindspore团队与北京大学董彬老师以及北京计算科学研究中心王艳莉老师合作，提出用稀疏神经表示求解玻尔兹曼方程。详见：[Solving Boltzmann equation with neural sparse representation](https://arxiv.org/abs/2302.09233)。样例代码请参考：[基于神经网络表示求解玻尔兹曼方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann)。
- 🔥`2023.02.05` [MindFlow 0.1.0-alpha](https://mindspore.cn/mindflow/docs/zh-CN/r0.1.0-alpha/index.html) 版本发布。
- 🔥`2023.01.17` 推出[MindFlow-CFD](https://zhuanlan.zhihu.com/p/599592997)基于MindSpore的端到端可微分求解器，[详见](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/mindflow/cfd)。
- 🔥`2022.12.27` MindSpore团队同西安交大陈刚老师合作发表[Temporal predictions of periodic flows using a mesh transformation and deep learning-based strategy](https://www.sciencedirect.com/science/article/pii/S1270963822007556)文章于航空领域Top期刊`Aerospace Science and Technology`，论文作者为邓志文、刘红升、时北极、王紫东、于璠、刘子扬(西交)、陈刚(通讯)。
- 🔥`2022.09.02` 中国商飞首席科学家吴光辉院士在WAIC2022世界人工智能大会发布首个工业级流体仿真大模型“东方.御风”, AI流体仿真助力国产大飞机气动仿真， [相关新闻](http://www.news.cn/fortune/2022-09/06/c_1128978806.htm)。

**更多应用案例请见**：👀

## Mindflow特性

- [基于MindFlow求解PINNs问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/features/solve_pinns_by_mindflow)

## 应用案例

### 物理驱动

|        案例            |        数据集               |    模型架构       |  GPU    |  NPU  |
|:----------------------:|:--------------------------:|:---------------:|:-------:|:------:|
|[PINNs求解Burgers方程](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/physics_driven/burgers)     |            [Burgers数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/)              |     PINNs        |   ✔️     |   ✔️   |
|[PINNs求解圆柱绕流流场](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/physics_driven/cylinder_flow)      |             [圆柱绕流流场数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/flow_past_cylinder/)              |        PINNs     |     ✔️   |   ✔️   |
|[PINNs求解Darcy流动](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/physics_driven/darcy)         |             -              |      PINNs      |  ✔️      |  ✔️    |
|[PINNs求解泊松方程](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/physics_driven/poisson)          |             -              |        PINNs     |  ✔️      |   ✔️   |
|[PINNs求解玻尔兹曼方程](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann)      |             -              |      PINNs       |   ✔️     |   ✔️   |
|[PINNs求解泰勒-格林涡](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/physics_driven/taylor_green/2d)      |             -              |      PINNs        |   ✔️     |   ✔️   |
|[PINNs求解NS方程反问题](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes_inverse)      |             [NS方程反问题数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/inverse_navier_stokes/)              |       PINNs       |   ✔️     |   ✔️   |

### 数据驱动

|        案例            |        数据集               |    模型架构       |  GPU    |  NPU  |
|:----------------------:|:--------------------------:|:---------------:|:-------:|:------:|
|[东方.御风](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_steady)   |  [二维翼型流场数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/airfoil/2D_steady/)   |     ViT           |   ✔️     |   ✔️   |
|[FNO求解Burgers方程](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/data_driven/burgers_fno)   | [一维Burgers方程数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/)     |     FNO1D       |   ✔️     |   ✔️   |
|[KNO求解Burgers方程](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/data_driven/burgers_kno)     | [一维Burgers方程数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/)       |       KNO1D       |   ✔️     |   ✔️   |
|[FNO求解NS方程](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes_fno)   |  [二维NS方程数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)         |        FNO2D          | ✔️   |   ✔️    |
|[FNO3d求解NS方程](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes_3d_fno)  | [二维NS方程数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)          |          FNO3D        |   ✔️     |   ✔️   |
|[KNO求解NS方程](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes_kno)    |  [二维NS方程数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)         |        KNO2D          |   ✔️     |   ✔️   |
|[CAE-LSTM求解二维黎曼问题](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/data_driven/cae_lstm/riemann)       |  [二维黎曼问题数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/riemann/)       |     CAE-LSTM      |   ✔️     |   ✔️   |
|[CAE-LSTM求解shu-osher](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/data_driven/cae_lstm/shu_osher)     |   [一维shu-osher波数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/shu_osher/)    |      CAE-LSTM      |   ✔️     |   ✔️   |
|[CAE-LSTM求解sod激波管问题](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/data_driven/cae_lstm/sod)   |  [一维sod激波管数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/sod/)         |     CAE-LSTM    |   ✔️     |   ✔️   |
|[CAE-LSTM求解KH问题](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/data_driven/cae_lstm/kh)         |  [二维K-H问题数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/kh/)            |  CAE-LSTM     |   ✔️     |   ✔️   |
|[ehdnn求解抖振流场](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/data_driven/transonic_buffet_ehdnn)          |  [二维翼型抖振数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/airfoil/2D_unsteady/)           |      ehdnn    |   ✔️     |   ✔️   |

### 数据-机理融合驱动

|          案例              |        数据集               |    模型架构       |  GPU    |  NPU  |
|:-------------------------:|:--------------------------:|:---------------:|:-------:|:------:|
| [PDE-NET求解对流扩散方程](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/variant_linear_coe_pde_net)   | 对流-扩散方程数据集()           |    PDE-Net    |   ✔️     |   ✔️   |

### CFD

|   案例        |     格式      |    GPU    |    NPU |
|:------------:|:-------------:|:---------:|:-------|
|[sod激波管](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/cfd/sod)      |    Rusanov    |       ✔️   |   -   |
|[lax激波管](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/cfd/lax)      |    Rusanov    |      ✔️    |   -   |
|[二维黎曼问题](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/cfd/riemann2d)    |       -       |     ✔️     |   -  |
|[库埃特流动](https://gitee.com/b_rookie/mindscience/tree/master/MindFlow/applications/cfd/couette)      |       -       |  ✔️        |   -   |

## **安装教程**

### 版本依赖关系

由于MindFlow与MindSpore有依赖关系，请根据下表中所指示的对应关系，在[MindSpore下载页面](https://www.mindspore.cn/versions)下载并安装对应的whl包。

| MindFlow |                                   分支                 |  MindSpore  | Python |
|:--------:|:----------------------------------------------------------------------:|:-----------:|:------:|
|  master  | [master](https://gitee.com/mindspore/mindscience/tree/master/MindFlow) |        \       | \>=3.7 |
| 0.1.0rc1 | [r0.2.0](https://gitee.com/mindspore/mindscience/tree/r0.2.0/MindFlow) |   \>=2.0.0rc1  | \>=3.7 |

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

# GPU version
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/MindScience/gpu/x86_64/cuda-11.1/mindflow_gpu-0.1.0rc1-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Ascend version
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0rc1/MindScience/ascend/aarch64/mindflow_ascend-0.1.0rc1-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
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

### 加入MindFlow SIG

<!DOCTYPE html>
<html lang="zh">
<head>
    <meta http-equiv="Content-Type" content="text/html" charset="utf-8">
</head>
<body>

<table id="t2" style="text-align:center" align="center">
    <tr id="tr2">
        <td>
            <img src="docs/co-chairs/张伟伟.jpeg" width="200" height="243"/>
            <p align="center">
                西北工业大学 张伟伟
            </p>
        </td>
        <td>
            <img src="docs/co-chairs/董彬.jpeg" width="200" height="243"/>
            <p align="center">
                北京大学 董彬
            </p>
        </td>
        <td>
            <img src="docs/co-chairs/孙浩.jpeg" width="200" height="243"/>
            <p align="center">
                中国人民大学 孙浩
            </p>
        </td>
    </tr>
</table>
</body>
</html>

[加入](https://mp.weixin.qq.com/s/e00lvKx30TsqjRhYa8nlhQ)昇思[MindFlow SIG](https://mindspore.cn/community/SIG/detail/?name=mindflow%20SIG)，助力AI流体仿真发展。
MindSpore AI+科学计算专题，北京大学董彬老师[Learning and Learning to solve PDEs](https://www.bilibili.com/video/BV1ur4y1H7vB?p=4)专题报告。
我们将不断发布[开源实习任务](https://gitee.com/mindspore/community/issues/I55B5A?from=project-issue)，与各位共同构筑MindFlow生态，与领域内的专家、教授和学生一起推动计算流体力学的发展，欢迎各位积极认领。

### 核心贡献者

感谢以下开发者做出的贡献 🧑‍🤝‍🧑：

yufan, wangzidong, liuhongsheng, zhouhongye, zhangyi, dengzhiwen, liulei, libokai, yangge, longzichao, qiuyisheng, haojiwei, leiyixiang

### 合作伙伴

<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
</head>
<body>
<table id="t1" style="text-align:center" align="center">
    <tr id="tr1">
        <td>
            <img src="docs/partners/CACC.jpeg"/>
            <p align="center">
                中国商飞
            </p>
        </td>
        <td>
            <img src="docs/partners/TaiHuLaboratory.jpeg"/>
            <p align="center">
                太湖实验室
            </p>
        </td>
        <td>
            <img src="docs/partners/NorthwesternPolytechnical.jpeg"/>
            <p align="center">
                西北工业大学
            </p>
        </td>
        <td>
            <img src="docs/partners/Peking_University.jpeg"/>
            <p align="center">
                北京大学
            </p>
        </td>
        <td>
            <img src="docs/partners/RenminUniversity.jpeg"/>
            <p align="center">
                中国人民大学
            </p>
        </td>
    </tr>
</table>
</body>
</html>

## **贡献指南**

- 如何贡献您的代码，请点击此处查看：[贡献指南](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/CONTRIBUTION.md)

## **许可证**

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
