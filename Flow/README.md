# 计算流体力学套件

## 概述

计算流体力学套件基于国产深度学习框架MindSpore开发，提供了一系列流体力学问题的AI求解方案。套件涵盖物理信息神经网络(PINNs)和数据驱动的神经算子两大类方法，适用于从基础方程求解到工业应用的多种场景。

## 应用案例

### 1. PINNs求解Burgers方程

**路径**: `burgers/`

**功能描述**: 使用物理信息神经网络求解一维Burgers方程。Burgers方程是一个非线性对流扩散方程，常用于测试流体力学算法。采用MultiScaleFCSequential网络，结合PDE损失、初始条件和边界条件进行训练。

**主要文件**:
- `burgers.py` - 主程序入口
- `configs/burgers.yaml` - 配置文件
- `src/dataset.py` - 数据集处理
- `src/model.py` - 网络模型

**支持平台**: Ascend/GPU/CPU

---

### 2. 翼型流场仿真（东方·御风）

**路径**: `2D_steady/`

**功能描述**: 工业级AI流场仿真模型"东方·御风"，用于预测超临界翼型周围的流场。基于ViT（Vision Transformer）架构，实现从翼型几何和气动参数到流场物理量的端到端映射。仿真时间缩短至传统CFD的1/24，平均误差降至1e-4量级。

**主要文件**:
- `2d_steady.py` - 主程序入口
- `configs/vit.yaml` - 配置文件
- `src/dataset.py` - 数据集处理
- `src/utils.py` - 工具函数
- `src/visualization.py` - 可视化模块

**关键特性**:
- 支持不同攻角、马赫数和翼型几何变化
- 使用小波变换损失函数提高激波区域预测精度
- 高效推理，适用于工业设计优化

---

### 3. PINNs求解圆柱绕流

**路径**: `cylinder_flow_forward/`

**功能描述**: 使用PINNs求解二维不可压缩Navier-Stokes方程，模拟圆柱绕流问题。采用多任务学习加权损失(MTLWeightedLoss)处理多个边界条件，求解时变流场。

**主要文件**:
- `cylinder_flow.py` - 主程序入口
- `configs/cylinder_flow.yaml` - 配置文件
- `src/dataset.py` - 数据集处理
- `src/model.py` - 网络模型

---

### 4. PINNs求解Darcy流动

**路径**: `darcy/`

**功能描述**: 使用PINNs求解二维Darcy方程，模拟多孔介质中的流动问题。采用FCSequential网络，结合PDE残差和边界条件进行训练，支持不规则几何区域。

**主要文件**:
- `darcy.py` - 主程序入口
- `configs/darcy.yaml` - 配置文件
- `src/darcy.py` - Darcy方程定义
- `src/dataset.py` - 数据集处理

---

### 5. PINNs求解玻尔兹曼方程

**路径**: `boltzmann/`

**功能描述**: 使用PINNs求解玻尔兹曼方程，支持多种碰撞模型（BGK、FBGK、FSM、LR、LA）。用于模拟稀薄气体动力学问题，可调节Knudsen数。

**主要文件**:
- `train.py` - 主程序入口
- `config/WaveD1V3_BGK.yaml` - 配置文件
- `boltzmann.py` - 玻尔兹曼方程求解
- `cells.py` - 网络单元
- `dataset.py` - 数据集处理

**关键特性**:
- 支持5种碰撞模型
- 使用SplitNet网络架构
- 包含速度空间离散化

---

### 6. PINNs求解泊松方程

**路径**: `poisson_continues/`, `Possion_21pinns/`

**功能描述**: 使用PINNs求解泊松方程，支持多种几何形状（矩形、圆盘、五边形、多边形、四面体、圆柱、圆锥等）和多种边界条件（Dirichlet、Periodic、Robin）。

**主要文件**:
- `poisson_continues/poisson_continues.py` - 连续版本主程序
- `poisson_continues/configs/poisson_cfg.yaml` - 配置文件
- `Possion_21pinns/` - 21种几何和边界条件组合

**关键特性**:
- 支持2D和3D问题
- 21种几何和边界条件组合
- 使用OneCycleLR学习率调度
- 包含符号微分定义PDE

---

## 扩展应用

### 7. PINNs求解圆柱绕流反问题

**路径**: `cylinder_flow_inverse/`

**功能描述**: 使用PINNs求解圆柱绕流的反问题，从观测数据反演流体参数（如粘性系数）。

---

### 8. PINNs求解Taylor Green涡流

**路径**: `taylor_green/`

**功能描述**: 使用PINNs求解二维Taylor-Green涡流问题，这是一个经典的湍流基准测试案例。

---

### 9. PINNs求解Kovasznay流动

**路径**: `kovasznay/`

**功能描述**: 使用PINNs求解Kovasznay流动，这是一个二维粘性流动的解析解问题，常用于验证算法准确性。

---

### 10. PINNs求解Allen-Cahn方程

**路径**: `allen_cahn/`

**功能描述**: 使用PINNs求解Allen-Cahn方程，这是一个反应扩散方程，用于模拟相分离过程。

---

## 神经算子方法

### 11. FNO求解Burgers方程（1D/2D/3D）

**路径**: `fno1d/`, `fno2d/`, `fno3d/`

**功能描述**: 使用傅里叶神经算子(FNO)求解Burgers方程和Navier-Stokes方程，学习从初始条件到解的映射。

---

### 12. KNO求解Burgers/Navier-Stokes方程

**路径**: `kno1d/`, `kno2d/`

**功能描述**: 使用Koornwinder神经算子(KNO)求解流体力学方程，采用正交多项式基函数。

---

### 13. SNO求解Burgers方程

**路径**: `sno1d/`

**功能描述**: 使用谱神经算子(SNO)求解一维Burgers方程，支持多种正交多项式（Legendre、Chebyshev等）。

---

## 技术特点

1. **技术路线完整**: 涵盖PINNs（物理驱动）和神经算子（数据驱动）两大类方法
2. **应用场景广泛**: 从基础方程到工业应用（翼型仿真）
3. **代码结构规范**: 每个应用包含configs、src、dataset等标准目录
4. **支持多硬件平台**: 均支持Ascend、GPU、CPU
5. **文档完善**: 主要应用都有README文档和配置文件

## 数据集与模型权重

### 数据集下载

| 应用 | 数据集下载地址 | 说明 |
|------|---------------|------|
| 2D_steady (翼型流场仿真) | 外部链接下载 | 创建dataset目录 |
| FNO求解Navier-Stokes方程 | https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/dataset/ | 放在fno2d/dataset目录下 |
| KNO求解Navier-Stokes方程 | https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes/ | 放在kno2d/dataset目录下 |
| FNO求解3D Navier-Stokes方程 | https://download.mindspore.cn/mindscience/mindflow/dataset/applications/data_driven/navier_stokes_3d/ | 放在fno3d/dataset目录下 |

### 模型权重下载

| 应用 | ckpt下载地址 |
|------|-------------|
| 2D_steady (翼型流场仿真) | https://onebox.huawei.com/#eSpaceGroupFile/1/41/13015984 |

## 运行环境

- MindSpore >= 2.0
- Python >= 3.7
- NumPy, SciPy, Matplotlib

## 使用方式

各应用目录下运行对应的Python脚本，例如：

```bash
# 运行Burgers方程求解
cd burgers
python burgers.py --config_path configs/burgers.yaml

# 运行翼型流场仿真
cd 2D_steady
python 2d_steady.py --config_path configs/vit.yaml
```
