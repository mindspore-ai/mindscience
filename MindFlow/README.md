[ENGLISH](README_EN.md) | 简体中文

[![master](https://img.shields.io/badge/version-master-blue.svg?style=flat?logo=Gitee)](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/README.md)
[![docs](https://img.shields.io/badge/docs-master-yellow.svg?style=flat)](https://mindspore.cn/mindflow/docs/zh-CN/master/index.html)
[![internship](https://img.shields.io/badge/internship-tasks-important.svg?style=flat)](https://gitee.com/mindspore/community/issues/I55B5A?from=project-issue)
[![SIG](https://img.shields.io/badge/community-SIG-yellowgreen.svg?style=flat)](https://www.mindspore.cn/community/SIG/detail/?name=mindflow+SIG)
[![Downloads](https://static.pepy.tech/badge/mindflow-gpu)](https://pepy.tech/project/mindflow-gpu)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://gitee.com/mindspore/mindscience/pulls)
[![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat)](https://github.com/mindspore-ai/mindspore/blob/master/LICENSE)

# **MindFlow**

## **MindFlow介绍**

流体仿真是指通过数值计算对给定边界条件下的流体控制方程进行求解，从而实现流动的分析、预测和控制，其在航空航天、船舶制造以及能源电力等行业领域的工程设计中应用广泛。传统流体仿真的数值方法如有限体积、有限差分等，主要依赖商业软件实现，需要进行物理建模、网格划分、数值离散、迭代求解等步骤，仿真过程较为复杂，计算周期长。AI具备强大的学习拟合和天然的并行推理能力，可以有效地提升流体仿真效率。

MindFlow是基于[昇思MindSpore](https://www.mindspore.cn/)开发的流体仿真领域套件，支持航空航天、船舶制造以及能源电力等行业领域的AI流场模拟，旨在于为广大的工业界科研工程人员、高校老师及学生提供高效易用的AI计算流体仿真软件。

<div align=center><img src="docs/mindflow_archi_cn.png" alt="MindFlow Architecture" width="700"/></div>

## **最新消息**

- 🔥`2024.03.22` 以“为智而昇，思创之源”为主题的昇思人工智能框架峰会2024在北京国家会议中心召开，北京国际数学研究中心教授、国际机器学习研究中心副主任董彬介绍，基于MindSpore和MindFLow套件，团队打造了AI解偏微分方程领域的基础模型PDEformer-1，能够直接接受任意形式PDE作为输入，通过在包含300万条一维PDE样本的庞大数据集上进行训练，PDEformer-1展现出了对广泛类型的一维PDE正问题的迅速且精准求解能力。
- 🔥`2024.03.22`以“为智而昇，思创之源”为主题的昇思人工智能框架峰会2024在北京国家会议中心召开，中国科学院院士、中国空气动力学会理事长唐志共介绍，基于昇思MindSpore和MindFlow套件，团队首创了生成式气动设计大模型平台，面向多种应用场景，打破传统设计范式，将设计时长由月级缩短到分钟级，满足概念设计要求[相关新闻](https://tech.cnr.cn/techph/20240323/t20240323_526636454.shtml)。
- 🔥`2024.03.20` MindFlow 0.2.0版本发布，详见[MindFlow 0.2.0](RELEASE_CN.md)。
- 🔥`2023.11.04`中国(西安)人工智能高峰论坛在西安市雁塔区高新国际会议中心召开，由西北工业大学与华为联合研发的首个面向飞行器的流体力学大模型“秦岭·翱翔”正式发布。该模型是西工大流体力学智能化国际联合研究所携手华为AI4Sci Lab在国产开源流体计算软件风雷的基础上，依托昇腾AI澎湃算力及昇思MindSpore AI框架共同研发的面向飞行器流体仿真的智能化模型，[相关新闻](https://mp.weixin.qq.com/s/Rhpiyf3VJYm_lMBWTRDtGA)。
- 🔥`2023.08.02` MindFlow 0.1.0版本发布，详见[MindFlow 0.1.0](https://mindspore.cn/mindflow/docs/zh-CN/r0.1/index.html)。
- 🔥`2023.07.06` 以“智联世界 生成未来”为主题的2023世界人工智能大会在上海世博中心开幕，来自中国商用飞机有限责任公司上海飞机设计研究院的三维超临界机翼流体仿真重器“东方.翼风”获得世界人工智能大会最高奖项——SAIL奖，该模型是由中国商用飞机有限责任公司上海飞机设计研究院与华为基于国产昇腾AI基础软硬件平台及昇思MindSpore AI框架研发的面向机翼复杂流动仿真场景的智能化模型，[相关新闻](https://www.thepaper.cn/newsDetail_forward_23769936)。
- 🔥`2023.05.21` 智能流体力学产业联合体第二次全体会议在杭州西湖大学成功举办，昇思MindSpore协办本次会议，三位中国科学院院士、产业联合体代表及关心联合体的学术界、产业界专家共计百位嘉宾现场参会。面向飞行器的首个流体力学大模型————“秦岭·翱翔”大模型预发布，该模型是由西北工业大学流体力学智能化国际联合研究所与华为基于国产昇腾AI基础软硬件平台及昇思MindSpore AI框架，共同研发的面向飞行器流体仿真的智能化模型，[相关新闻](http://science.china.com.cn/2023-05/23/content_42378458.htm)。
- 🔥`2023.02.05` [MindFlow 0.1.0-alpha](https://mindspore.cn/mindflow/docs/zh-CN/r0.1.0-alpha/index.html) 版本发布。
- 🔥`2023.01.17` 推出[MindFlow-CFD](https://zhuanlan.zhihu.com/p/599592997)基于MindSpore的端到端可微分求解器，[详见](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/mindflow/cfd)。
- 🔥`2022.09.02` 中国商飞首席科学家吴光辉院士在WAIC2022世界人工智能大会发布首个工业级流体仿真大模型“东方.御风”, AI流体仿真助力国产大飞机气动仿真， [相关新闻](http://www.news.cn/fortune/2022-09/06/c_1128978806.htm)。

## 论文

Ye Z, Huang X, Liu H, et al. Meta-Auto-Decoder: A Meta-Learning Based Reduced Order Model for Solving Parametric Partial Differential Equations[J]. Communications on Applied Mathematics and Computation. [[Paper]](https://link.springer.com/article/10.1007/s42967-023-00293-7)

Deng Z, Wang J, Liu H, et al. Prediction of transactional flow over supercritical airfoils using geometric-encoding and deep-learning strategies. Physics of Fluids 35, 075146 (2023). [[Paper]](https://pubs.aip.org/aip/pof/article-abstract/35/7/075146/2903765/Prediction-of-transonic-flow-over-supercritical?redirectedFrom=fulltext)
[[Code]](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_steady)

Rao C, Ren P, Wang Q, et al. Encoding physics to learn reaction–diffusion processes[J]. Nature Machine Intelligence, 2023: 1-15. [[Paper]](https://arxiv.org/abs/2106.04781)
[[Code]](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/percnn)

Li Z, Wang Y, Liu H, et al. Solving Boltzmann equation with neural sparse representation[J]. SIAM Journal on Scientific Computing, Vol. 46, Iss. 2 (2024).
[[Paper]](https://epubs.siam.org/doi/abs/10.1137/23M1558227?journalCode=sjoce3)
[[Code]](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann)

Deng Z, Liu H, Shi B, et al. Temporal predictions of periodic flows using a mesh transformation and deep learning-based strategy[J]. Aerospace Science and Technology, 2023, 134: 108081. [[Paper]](https://www.sciencedirect.com/science/article/pii/S1270963822007556)

Huang X, Liu H, Shi B, et al. A Universal PINNs Method for Solving Partial Differential Equations with a Point Source[C]//IJCAI. 2022: 3839-3846. [[Paper]](https://gitee.com/link?target=https%3A%2F%2Fwww.ijcai.org%2Fproceedings%2F2022%2F0533.pdf) [[Code]](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/point_source)

## 特性

- [MindSpore自动微分详解](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/features/mindspore_grad_cookbook.ipynb)

- [基于MindFlow求解PINNs问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/features/solve_pinns_by_mindflow)

## 应用案例

### 数据驱动

|        案例            |        数据集               |    模型架构       |  GPU    |  NPU  |
|:----------------------:|:--------------------------:|:---------------:|:-------:|:------:|
|[东方.御风](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_steady)   |  [二维翼型流场数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/airfoil/2D_steady/)   |     ViT           |   ✔️     |   ✔️   |
|[FNO方法求解Burgers方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers/fno1d)   | [一维Burgers方程数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/)     |     FNO1D       |   ✔️     |   ✔️   |
|[KNO方法求解Burgers方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/burgers/kno1d)     | [一维Burgers方程数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/burgers/)       |       KNO1D       |   ✔️     |   ✔️   |
|[FNO方法求解NS方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/fno2d)   |  [二维NS方程数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)         |        FNO2D          | ✔️   |   ✔️    |
|[FNO3D方法求解NS方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/fno3d)  | [二维NS方程数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)          |          FNO3D        |   ✔️     |   ✔️   |
|[KNO方法求解NS方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/navier_stokes/kno2d)    |  [二维NS方程数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/)         |        KNO2D          |   ✔️     |   ✔️   |
|[CAE-LSTM方法求解二维黎曼问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)       |  [二维黎曼问题数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/riemann/)       |     CAE-LSTM      |   ✔️     |   ✔️   |
|[CAE-LSTM方法求解Shu-Osher问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)     |   [一维Shu-Osher波数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/shu_osher/)    |      CAE-LSTM      |   ✔️     |   ✔️   |
|[CAE-LSTM方法求解Sod激波管问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)   |  [一维Sod激波管数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/sod/)         |     CAE-LSTM    |   ✔️     |   ✔️   |
|[CAE-LSTM方法求解KH问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_lstm)         |  [二维K-H问题数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-lstm/kh/)            |  CAE-LSTM     |   ✔️     |   ✔️   |
|[eHDNN方法求解抖振流场](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/transonic_buffet_ehdnn)          |  [二维翼型抖振数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/airfoil/2D_unsteady/)           |      eHDNN    |   ✔️     |   ✔️   |
|[eHDNN方法预测非定常流场](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/move_boundary_hdnn)          |  [动边界流场数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/move_boundary_hdnn)           |      eHDNN    |   ✔️     |   ✔️   |
|[ResUnet3D方法求解三维圆球绕流](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/flow_around_sphere)          |  [三维非定常流动数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/3d_unsteady_flow/)           |      ResUnet3D    |   ✔️     |   ✔️   |
|[CAE-Transformer方法求解二维圆柱绕流问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cae_transformer)          |  [低雷诺数圆柱绕流数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/cae-transformer/)           |      CAE-Transformer    |   ✔️     |   ✔️   |
|[FNO2D和UNET2D方法预测多时间步跨声速翼型复杂流场](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_driven/airfoil/2D_unsteady)          |  [二维跨声速翼型复杂流场数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/airfoil/2D_unsteady/)           |      FNO2D/UNET2D    |   ✔️     |   ✔️   |
|[HDNN方法预测流固耦合系统流场](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/fluid_structure_interaction)          |  [流固耦合系统数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/fluid_structure_interaction/)           |      HDNN    |   ✔️     |   ✔️   |

### 数据-机理融合驱动

|          案例              |        数据集               |    模型架构       |  GPU    |  NPU  |
|:-------------------------:|:--------------------------:|:---------------:|:-------:|:------:|
| [PDE-NET方法求解对流扩散方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/pde_net)   | -           |    PDE-Net    |   ✔️     |   ✔️   |
|   [PeRCNN方法求解二维Burgers方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/percnn/burgers_2d)  |    [PeRCNN数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN/) |  PeRCNN  |   ✔️     |   ✔️   |
|   [PeRCNN方法求解三维反应扩散方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/percnn/gsrd_3d)  |    [PeRCNN数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PeRCNN/) |  PeRCNN  |   ✔️     |   ✔️   |
| [AI湍流模型](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/data_mechanism_fusion/ai_turbulence_modeling)   | -           |    MLP    |   ✔️     |   ✔️   |

### 物理驱动

|        案例            |        数据集               |    模型架构       |  GPU    |  NPU  |
|:----------------------:|:--------------------------:|:---------------:|:-------:|:------:|
|[PINNs方法求解Burgers方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/burgers)     |            [Burgers数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/burgers_pinns/)              |     PINNs        |   ✔️     |   ✔️   |
|[PINNs方法求解圆柱绕流](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/cylinder_flow_forward)      |             [圆柱绕流流场数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/flow_past_cylinder/)              |        PINNs     |     ✔️   |   ✔️   |
|[PINNs方法求解Darcy流动](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/darcy)         |             -              |      PINNs      |  ✔️      |  ✔️    |
|[PINNs方法求解泊松方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/continuous)          |             -              |        PINNs     |  ✔️      |   ✔️   |
|[PINNs方法求解玻尔兹曼方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/boltzmann)      |             -              |      PINNs       |   ✔️     |   ✔️   |
|[PINNs方法求解泰勒-格林涡](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/taylor_green)      |             -              |      PINNs        |   ✔️     |   ✔️   |
|[PINNs方法求解NS方程反问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/cylinder_flow_inverse)      |             [NS方程反问题数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/physics_driven/inverse_navier_stokes/)              |       PINNs       |   ✔️     |   ✔️   |
|[PINNs方法求解二维带点源的泊松方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/poisson/point_source)      |             -              |       PINNs       |   ✔️     |   ✔️   |
|[PINNs方法求解Kovasznay流动](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/kovasznay)      |             -              |       PINNs       |   ✔️     |   ✔️   |
|[PINNs方法求解周期山流动问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/physics_driven/navier_stokes/periodic_hill)      |             [Periodic  Hill数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/applications/research/allen_cahn/)              |       PINNs       |   ✔️     |   ✔️   |
|[PINNs方法求解Allen-Cahn方程](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/allen_cahn)      |             [Allen-Cahn数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/periodic_hill_2d/)              |       PINNs       |   ✔️     |   ✔️   |
|[CMA-ES&多目标梯度下降算法](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/cma_es_mgda)      |             [Periodic Hill数据集](https://download.mindspore.cn/mindscience/mindflow/dataset/periodic_hill_2d/)              |       PINNs       |   ✔️     |   ✔️   |
|[META-PINNs算法](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/meta_pinns)         |             -              |      PINNs      |  ✔️      |  ✔️    |
|[MOE-PINNs算法](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/moe_pinns)         |             -              |      PINNs      |  ✔️      |  ✔️    |
|[R-DLGA算法](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/research/r_dlga)         |             -              |      PINNs      |  ✔️      |  ✔️    |

### CFD

|   案例        |     格式      |    GPU    |    NPU |
|:------------:|:-------------:|:---------:|:-------|
|[Sod激波管](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/sod)      |    Rusanov    |       ✔️   |   -   |
|[Lax激波管](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/lax)      |    Rusanov    |      ✔️    |   -   |
|[二维黎曼问题](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/riemann2d)    |       -       |     ✔️     |   -  |
|[库埃特流动](https://gitee.com/mindspore/mindscience/tree/master/MindFlow/applications/cfd/couette)      |       -       |  ✔️        |   -   |

## **安装教程**

### 版本依赖关系

由于MindFlow与MindSpore有依赖关系，请根据下表中所指示的对应关系，在[MindSpore下载页面](https://www.mindspore.cn/versions)下载并安装对应的whl包。

| MindFlow |                                   分支                 |  MindSpore  | Python |
|:--------:|:----------------------------------------------------------------------:|:-----------:|:------:|
|  master  | [master](https://gitee.com/mindspore/mindscience/tree/master/MindFlow) |        \       | \>=3.7 |
| 0.2.0  | [r0.6](https://gitee.com/mindspore/mindscience/tree/r0.6/MindFlow) |   \>=2.2.12  | \>=3.7 |
| 0.1.0    | [r0.3](https://gitee.com/mindspore/mindscience/tree/r0.3/MindFlow) |   \>=2.0.0  | \>=3.7 |
| 0.1.0rc1 | [r0.2.0](https://gitee.com/mindspore/mindscience/tree/r0.2.0/MindFlow) |   \>=2.0.0rc1  | \>=3.7 |

### 依赖安装

```bash
pip install -r requirements.txt
```

### 硬件支持情况

| 硬件平台       | 操作系统        | 状态 |
| :------------ | :-------------- | :--- |
| Ascend        | Linux            | ✔️ |
| GPU           | Linux           | ✔️ |

### pip安装

```bash
# gpu and ascend are supported
export DEVICE_NAME=gpu
pip install mindflow_${DEVICE_NAME}
```

### 源码安装

- 从Gitee下载源码。

```bash
git clone https://gitee.com/mindspore/mindscience.git
cd {PATH}/mindscience/MindFlow
```

- 编译Ascend后端源码。

```bash
bash build.sh -e ascend -j8
```

- 编译GPU后端源码。

```bash
export CUDA_PATH={your_cuda_path}
bash build.sh -e gpu -j8
```

- 安装编译所得whl包。

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
                <td>
            <img src="docs/co-chairs/马浩.jpeg" width="200" height="243"/>
            <p align="center">
                郑州航空工业管理学院 马浩
            </p>
        </td>
    </tr>
</table>

</body>
</html>

[加入](https://mp.weixin.qq.com/s/e00lvKx30TsqjRhYa8nlhQ)昇思[MindFlow SIG](https://www.mindspore.cn/community/SIG/detail/?name=mindflow+SIG)，助力AI流体仿真发展。
MindSpore AI+科学计算专题，北京大学董彬老师[Learning and Learning to solve PDEs](https://www.bilibili.com/video/BV1ur4y1H7vB?p=4)专题报告。
我们将不断发布[开源实习任务](https://gitee.com/mindspore/community/issues/I55B5A?from=project-issue)，与各位共同构筑MindFlow生态，与领域内的专家、教授和学生一起推动计算流体力学的发展，欢迎各位积极认领。

### 核心贡献者

感谢以下开发者做出的贡献 🧑‍🤝‍🧑：

yufan, wangzidong, liuhongsheng, zhouhongye, zhangyi, dengzhiwen, liulei, guoboqiang, chengzeruizhi, libokai, yangge, longzichao, qiuyisheng, haojiwei, leiyixiang, huangxiang, huxin,xingzhongfan, mengqinghe, lizhengyi, lixin, liuziyang, dujiaoxi, xiaoruoye, liangjiaming

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
        <td>
            <img src="docs/partners/HIT.jpeg"/>
            <p align="center">
                哈尔滨工业大学
            </p>
        </td>
    </tr>
</table>
</body>
</html>

## **贡献指南**

- 如何贡献您的代码，请点击此处查看：[贡献指南](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/CONTRIBUTION_CN.md)
- 需要算力的用户，请参考[启智社区云脑使用指南](https://download-mindspore.osinfra.cn/mindscience/mindflow/tutorials/%E5%90%AF%E6%99%BA%E6%8C%87%E5%8D%97.pdf), [NPU使用录屏](https://download-mindspore.osinfra.cn/mindscience/mindflow/tutorials/npu%E4%BD%BF%E7%94%A8.MP4), [GPU使用录屏](https://download-mindspore.osinfra.cn/mindscience/mindflow/tutorials/gpu%E4%BD%BF%E7%94%A8.MP4)

## **许可证**

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
