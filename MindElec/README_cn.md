# MindElec

[View English](README.md)

<!-- TOC --->

- [MindElec介绍](#mindelec介绍)
    - [数据构建及转换](#数据构建及转换)
    - [仿真计算](#仿真计算)
        - [AI电磁模型库](#AI电磁模型库)
        - [优化策略](#优化策略)
    - [结果可视化](#结果可视化)
- [安装教程](#安装教程)
    - [确认系统环境信息](#确认系统环境信息)
    - [pip安装](#pip安装)
        - [安装MindSpore](#安装mindspore)
        - [安装MindElec](#安装mindelec)
    - [源码安装](#源码安装)
- [API](#api)
- [验证是否成功安装](#验证是否成功安装)
- [快速入门](#快速入门)
- [文档](#文档)
- [社区](#社区)
    - [治理](#治理)
- [贡献](#贡献)
- [版本说明](#版本说明)
- [许可证](#许可证)

<!-- /TOC -->

## MindElec介绍

电磁仿真是指通过计算的方式模拟电磁波在物体或空间中的传播特性，其在手机容差、天线优化和芯片设计等场景中应用广泛。传统数值方法如有限差分、有限元等需网格剖分、迭代计算，仿真流程复杂、计算时间长，无法满足产品的设计需求。AI方法具有万能逼近能力和高效推理能力，可有效提升仿真效率。

MindElec是基于MindSpore开发的AI电磁仿真工具包，由数据构建及转换、仿真计算、以及结果可视化组成。可以支持端到端的AI电磁仿真。目前已在华为终端手机容差场景中取得阶段性成果，相比商业仿真软件，AI电磁仿真的S参数误差在2%左右，端到端仿真速度提升10+倍。

<div align=center>
<img src="docs/MindElec-architecture.jpg" alt="MindElec Architecture" width="600"/>
</div>

### 数据构建及转换

支持CSG （Constructive Solid Geometry，CSG）
模式的几何构建，如矩形、圆形等的交集、并集和差集等，以及cst和stp数据（CST等商业软件支持的数据格式）的高效张量转换。未来还会支持智能网格剖分，为传统科学计算使用。

### 仿真计算

#### AI电磁模型库

包括物理和数据驱动的AI电磁模型：物理驱动是指网络的训练无需额外的标签数据，只需方程和初边界条件即可；数据驱动是指训练需使用仿真或实验等产生的数据。物理驱动相比数据驱动，优势在于可避免数据生成带来的成本和网格独立性等问题，劣势在于需明确方程的具体表达形式并克服点源奇异性、多任务损失函数以及泛化性等技术挑战。

#### 优化策略

通过接收AI亲和的张量数据，利用物理和数据驱动的方式进行电磁仿真。为提升模型的精度、减少训练的成本，提供了一系列优化措施。数据压缩可以有效地减少神经网络的存储和计算量；多尺度滤波、动态自适应加权可以提升模型的精度，克服点源奇异性等问题；小样本学习主要是为了减少训练的数据量，节省训练的成本。

### 结果可视化

仿真的结果如S参数或电磁场等可保存在CSV、VTK文件中。MindInsight可以显示训练过程中的损失函数变化，并以图片的形式在网页上展示结果；Paraview是第三方开源软件，可动态展示切片、翻转等高级功能。

## 安装教程

### 确认系统环境信息

| 硬件平台      | 操作系统        | 状态  |
| :------------ | :-------------- | :--- |
| Ascend 910    | Ubuntu-x86      | ✔️   |
|               | Ubuntu-aarch64  | ✔️   |
|               | EulerOS-aarch64 | ✔️   |
|               | CentOS-x86      | ✔️   |
|               | CentOS-aarch64  | ✔️   |

- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装，要求至少1.4.0版本。
- 其余依赖请参见[requirements.txt](https://gitee.com/mindspore/mindsciencetmp/blob/master/MindElec/requirements.txt)

### pip安装

#### 安装MindSpore

```bash
pip install https://hiq.huaweicloud.com/download/mindspore/ascend/x86_64/mindspore-1.5.0-cp38-cp38-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 请根据本机的python版本选择合适的安装包，如本机为python 3.7，则可将上面命令中的`cp38-cp38`修改为`cp37-cp37`。

#### 安装MindElec

```bash
pip install https://hiq.huaweicloud.com/download/mindscience/x86_64/mindscience_mindelec_ascend-0.1.0-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindElec安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindsciencetmp/blob/master/MindElec/setup.py)），点云数据采样依赖[pythonocc](https://github.com/tpaviot/pythonocc-core), 需自行安装。

### 源码安装

1. 从代码仓下载源码

    ```bash
    cd ~
    git clone https://gitee.com/mindspore/mindscience.git
    ```

2. 编译安装MindElec

    ```bash
    cd ~/MindElec
    bash build.sh
    pip install output/mindscience_mindelec_ascend-{version}-cp37-cp37m-linux_{x86_64}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

## API

MindElec API文档请查看[文档链接](https://www.mindspore.cn/mindelec/api/zh-CN/master/index.html)

## 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindelec'`，则说明安装成功。

```bash
python -c 'import mindelec'
```

## 快速入门

关于如何快速使用AI电磁仿真工具包，进行训练推理，请点击查看[MindElec使用教程](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/index.html)

## 文档

有关安装指南、教程和API的更多详细信息，请参阅[用户文档](https://gitee.com/mindspore/docs)。

## 社区

### 治理

查看MindSpore如何进行[开放治理](https://gitee.com/mindspore/community/blob/master/governance.md)。

## 贡献

欢迎参与贡献。更多详情，请参阅我们的[贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

## 版本说明

版本说明请参阅[RELEASE](https://gitee.com/mindspore/mindscience/blob/master/MindElec/RELEASE.md)。

## 许可证

[Apache License 2.0](LICENSE)
