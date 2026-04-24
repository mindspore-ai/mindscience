# 电磁仿真套件

## 概述

电磁仿真套件基于国产深度学习框架MindSpore开发，提供了一系列电磁场问题的AI求解方案。套件涵盖物理信息神经网络(PINNs)和可微FDTD两大类方法，适用于时域/频域电磁场求解、电磁参数化仿真、电磁逆问题等应用场景。

## 应用案例

### 1. PINNs求解点源麦克斯韦方程

**路径**: `mindelec/time_domain_maxwell/`

**功能描述**: 使用PINNs（物理信息神经网络）求解时域麦克斯韦方程。采用TE模式波，包含二阶Mur吸收边界条件。使用MultiScaleFCCell多尺度网络，通过高斯脉冲源激励，求解电磁场在时间和空间上的分布。

**主要文件**:
- `time_maxwell.py` - 主程序入口
- `config.json` - 配置文件
- `src/` - 核心模块

**关键特性**:
- 求解2D Maxwell方程组
- 支持随机采样和网格采样
- 使用MTL加权损失函数
- 支持训练过程可视化

---

### 2. 可微FDTD电磁正向问题

**路径**: `mindelec/fdtd_forward/`

**功能描述**: 基于自动微分FDTD方法求解电磁正向问题。通过可微分的时域有限差分法，计算天线的S参数。支持完全3D仿真，包含CFS-PML边界条件，可处理微带滤波器等复杂结构。

**主要文件**:
- `solve_invert_f.py` - 求解主程序
- `dataset/` - 数据集
- `src/` - 核心模块

**关键特性**:
- 自动微分FDTD求解器
- 支持3D全波仿真
- 计算S11/S21等散射参数
- 包含电压源、电流监视器等组件

---

### 3. PINNs求解点源麦克斯韦方程族问题（频域）

**路径**: `mindelec/frequency_domain_maxwell/`

**功能描述**: 使用PINNs求解频域麦克斯韦方程（Helmholtz方程）。求解二维Helmholtz方程：∇²u + k²u = 0，其中k为波数。适用于频域电磁场问题，如谐振腔、波导等。

**主要文件**:
- `frequency_maxwell.py` - 主程序入口
- `src/helmholtz.py` - Helmholtz方程定义
- `src/config.py` - 配置模块

**关键特性**:
- 求解2D Helmholtz方程
- 支持自定义波数
- 包含边界条件约束
- 使用FFNN前馈神经网络

---

### 4. AI参数化电磁仿真

**路径**: `mindelec/parameterization/`

**功能描述**: 使用深度学习进行电磁参数化仿真。通过神经网络学习天线几何参数到S11参数的映射关系，实现快速电磁仿真。支持蝴蝶天线、手机天线等多种天线类型。

**主要文件**:
- `train.py` - 训练主程序
- `src/maxwell_model.py` - S11预测模型
- `dataset/` - 数据集

**关键特性**:
- 快速预测S11参数
- 支持多种天线类型
- 输入为几何参数（3维）
- 输出为频率响应（1001个频点）
- 使用残差连接的深度网络

---

### 5. 探地雷达电磁反演

**路径**: `mindelec/gprinversion/`

**功能描述**: 探地雷达（GPR）电磁反演应用。通过测量得到的电磁场数据（Ez、Hx、Hy），反演地下目标的位置信息。使用卷积神经网络处理时序电磁场数据，预测目标坐标。

**主要文件**:
- `train.py` - 训练主程序
- `eval.py` - 评估脚本
- `metric.py` - 评估指标
- `dataset/` - 数据集

**关键特性**:
- 输入：电磁场分量
- 输出：目标位置坐标（2维）
- 使用1D卷积网络
- 支持数据归一化处理
- 适用于地下目标探测

---

### 6. 可微FDTD电磁逆问题

**路径**: `mindelec/fdtd_inverse/`

**功能描述**: 基于自动微分FDTD的电磁逆散射问题求解。通过观测到的电磁场数据，反演介质的介电常数分布。使用拓扑优化方法，通过梯度下降优化材料参数。

**主要文件**:
- `solve.py` - 求解主程序
- `dataset/` - 数据集
- `src/` - 核心模块

**关键特性**:
- 自动微分FDTD逆问题求解器
- 反演介电常数分布
- 支持多源多接收器配置
- 使用ELU激活函数进行参数映射
- 适用于电磁逆散射成像

---

## 扩展应用（SciAI模型库）

`model_sciai/` 目录下包含16个PINNs变体模型，用于求解各类偏微分方程：

| 模型 | 功能描述 |
|------|---------|
| `auq_pinns` | 对抗不确定性量化PINNs |
| `cpinns` | 守恒PINNs |
| `deep_hpms` | 深度哈密顿模型 |
| `deep_ritz` | Deep Ritz方法 |
| `fpinns` | 分数阶PINNs |
| `gradient_pathologies_pinns` | 梯度病理PINNs（求解Helmholtz方程） |
| `hp_vpinns` | hp变分PINNs |
| `laaf` | 局部自适应激活函数 |
| `label_free_dnn_surrogate` | 无标签DNN代理 |
| `multiscale_pinns` | 多尺度PINNs |
| `nsf_nets` | Navier-Stokes流网络 |
| `pinn_heattransfer` | 热传导PINNs |
| `pinns_ntk` | 神经正切核PINNs |
| `pinns_swe` | 浅水波方程PINNs |
| `sympnets` | 辛网络 |
| `xpinns` | 扩展PINNs |

---

## 技术特点

1. **方法全面**: 涵盖PINNs和可微FDTD两大类方法
2. **应用广泛**: 支持正向问题、反问题、参数化仿真
3. **时频域兼顾**: 支持时域和频域电磁场求解
4. **工业应用**: 支持天线设计、探地雷达等实际应用
5. **可微分求解**: 基于自动微分的FDTD实现端到端优化

## 案例统计

- model_sciai目录: 16个PINNs变体案例
- mindelec目录: 6个电磁仿真案例

## 运行环境

- MindSpore >= 2.0
- MindElec
- Python >= 3.7
- NumPy, SciPy, Matplotlib

## 使用方式

各应用目录下运行对应的Python脚本，例如：

```bash
# 运行时域麦克斯韦方程求解
cd mindelec/time_domain_maxwell
python time_maxwell.py

# 运行FDTD正向问题
cd mindelec/fdtd_forward
python solve_invert_f.py

# 运行AI参数化仿真
cd mindelec/parameterization
python train.py
```
