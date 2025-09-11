# DeepONet-Grid-UQ

用于电力系统故障后进行预测的DeepONet-Grid网络

## 背景介绍

### 需求来源及价值概述

本工作构建了一个高效的网络DeepONet-Grid，用于对故障后的电力系统进行动态安全分析，该网络

(i) 接收故障前和故障期间收集的轨迹作为输入，并且
(ii) 输出预测的故障后轨迹。

此外，本网络还通过不确定性量化（Uncertainty Quantification）为其方法赋予了在效率与可靠/可信预测之间取得平衡的能力。

原始论文：[DeepONet-grid-UQ: A trustworthy deep operator framework for predicting the power grid's post-fault trajectories](!https://www.sciencedirect.com/science/article/abs/pii/S0925231223002503)

原始代码仓：[Github Link](!https://github.com/cmoyacal/DeepONet-Grid-UQ)

### 研究背景与动机

电力系统作为关键基础设施，其稳定性和可靠性对现代社会至关重要。然而，电网经常面临罕见但严重的故障和扰动，这些事件可能导致系统不稳定，甚至引发大规模停电。

传统的动态安全分析需要求解复杂的非线性微分代数方程组，计算成本极高，难以实现实时分析。随着电网的转型，电力公司迫切需要能够进行近实时的动态安全评估。

现有的机器学习方法主要关注二分类问题（稳定/不稳定），缺乏对故障后轨迹的定量预测能力。系统运营商和规划者需要了解故障后各种状态变量的轨迹，以评估电压或频率是否会违反预定义限制并触发负荷切除等保护措施。

## 项目结构

```bash
deeponet-grid/
├── configs/
│   └── config.yaml          # 配置文件
├── src/
│   ├── model.py             # DeepONet模型定义
│   ├── data.py              # 数据加载和预处理
│   ├── utils.py             # 一些有用的函数
│   ├── trainer.py           # 训练器实现
│   └── metrics.py           # 评估指标
├── train.py                 # 训练脚本
├── inference.py             # 推理脚本
├── requirements.txt         # 依赖包列表
├── README.md                # 本文件
├── README_en.md             # 英文版本
└── outputs/                 # 输出目录（自动创建）
```

## 安装与配置

### 安装MindSpore

安装MindSpore框架：

```bash
pip install mindspore
```

### 配置文件

编辑 `configs/config.yaml` 文件来配置模型参数：

#### 模型配置

- `branch`: 分支网络配置（处理输入函数）
- `trunk`: 主干网络配置（处理评估点）
- `use_bias`: 是否使用偏置项

#### 训练配置

- `learning_rate`: 学习率
- `batch_size`: 批次大小
- `epochs`: 训练轮数
- `optimizer`: 优化器类型（adam）
- `loss_type`: 损失函数类型（nll, mse）

#### 数据配置

- `data_path`: 数据文件路径

## 使用方法

### 1. 使用真实数据训练

可以使用数据集: [dataset](https://download.mindspore.cn/mindscience/mindenergy/dataset/applications/DeepONet-grid/). 感谢数据集提供者: lzh9673@163.com.

首先确保`confis/config.yaml`文件中的`data_path`参数指向训练数据的地址。

```bash
python train.py
```

多卡：

```bash
msrun -worker_num 8 python train.py --distributed 1
```

#### 输出文件

训练完成后，在输出目录中会生成：

- `best_model.ckpt`: 最佳模型检查点
- `final_model.ckpt`: 最终模型检查点
- `training_history.json`: 训练历史记录
- `test_results.json`: 测试集评估结果
- `training.log`: 训练日志

#### 评估指标

- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **R²**: 决定系数
- **Calibration Error**: 校准误差

### 2. 从检查点恢复训练

```bash
python train.py --resume outputs/best_model.ckpt
```

### 3. 仅运行评估

```bash
python train.py --eval --resume outputs/best_model.ckpt
```

### 4. 打印loss log曲线

执行`src/utils.py`中的`extract_log`函数，并将记录训练信息和评估信息的地址传入。

### 5. 仅运行推理

```bash
# 指定数据推理
python inference.py --checkpoint outputs/best_model.ckpt \
                   --data_path data/test-data-voltage-m-33-mix.npz \
                   --trajectory_prediction \
                   --data_index 0

# 数据集推理
python inference.py --checkpoint outputs/best_model.ckpt \
                   --data_path data/test-data-voltage-m-33-mix.npz \
                   --output_dir inference_results
```

### 6. 调试模式

设置环境变量启用详细日志：

```bash
export MINDSPORE_LOG_LEVEL=DEBUG
python train.py
```

## 扩展内容
### 模型架构

DeepONet由两个主要组件组成：

1. **分支网络 (Branch Network)**: 处理输入函数 `u(x)`
2. **主干网络 (Trunk Network)**: 处理评估点 `y`

输出通过点积计算：
```
G(u)(y) = Σᵢ bᵢ(u) tᵢ(y)
```

其中 `bᵢ(u)` 是分支网络的输出，`tᵢ(y)` 是主干网络的输出。

### 不确定性量化

模型提供不确定性量化功能：

- **均值预测**: 模型预测的期望值
- **标准差预测**: 预测的不确定性度量

损失函数支持：
- **负对数似然 (NLL)**: 用于不确定性量化
- **均方误差 (MSE)**: 标准回归损失


### 数据格式

数据文件为 `.npz` 格式，包含以下字段：

- `u`: 输入函数值，形状为 `(n_samples, n_sensors)`
- `y`: 评估点，形状为 `(n_samples, n_points, n_dim)`
- `s`: 真实解值，形状为 `(n_samples, n_points, n_output)`

## 训练结果

|     参数      |     指标      |
| :-----------: | :-----------: |
|   硬件资源    | Atlas 800T A2 |
| MindSpore版本 |    >=2.5.0    |
|    数据量     |     80000     |
|   训练步数    |     1000      |
|   Scheduler   |    Cosine     |
|  Batch Size   |     1024      |
|  最小学习率   |     1e-7      |
|  最大学习率   |     5e-5      |
|      MSE      |   0.003331    |
|      MAE      |   0.024189    |
|      L1       |   0.024222    |
|      L2       |   0.057664    |