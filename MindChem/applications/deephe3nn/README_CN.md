# DeephE3nn

## 背景简介

DeephE3nn 是一个基于 E(3) 等变神经网络的模型，用于从晶体中的原子构型精准预测体系的电子哈密顿量。在传统第一性原理计算中，每一个原子结构都需要通过密度泛函理论等高成本方法重新求解哈密顿量，计算代价随体系规模快速增长，难以支撑大规模材料筛选与高通量模拟。

借助等变图神经网络对空间旋转和平移对称性的显式建模，DeephE3nn 能够高效学习「晶体结构 → 电子哈密顿量」的映射关系，在保证物理对称性的前提下显著降低计算成本。本示例基于双层石墨烯数据集，对材料体系的电子哈密顿量进行预测，可为后续能带结构、输运性质等下游任务提供高效近似。

## 模型实现

### 硬件要求

- 当前脚本默认在 `Ascend` 设备上运行，通过命令行参数 `-device_id` 指定卡号（见 `train.py` 与 `predict.py`）。

### 版本依赖

- 需要安装 `MindSpore >= 2.7.0`。
- 需要安装 `MindScience >= 0.8.0`，以提供 `mindscience.e3nn` 等相关模块。

### 安装

- 安装 MindSpore：参考官方安装指南 `https://www.mindspore.cn/install`
- 安装 MindScience：参考 `https://atomgit.com/mindspore-lab/mindscience`
- 安装依赖包：`pip install -r requirements.txt`

### 数据集

- 从 [Zenodo 数据集页面](https://zenodo.org/records/7553640) 下载 `Bilayer_graphene_dataset.zip` 到当前目录并解压，保持文件名不变。

解压后目录结构示例（仅示意）：

```txt
deephe3nn
    ├─Bilayer_graphene_dataset
    │      ...
    └─configs
           Bilayer_graphene_train.ini
```

### 核心代码实现

- 代码主要模块位于 `data`、`graph` 与 `models` 文件夹，并依赖 MindScience 中的 `mindscience.e3nn` 等模块：

```text
applications
  └── deephe3nn
        ├── README.md                     # 中文说明
        ├── README_EN.md                  # 英文说明
        ├── train.py                      # 训练入口
        ├── predict.py                    # 推理入口
        ├── requirements.txt              # 环境依赖（第三方 Python 包）
        ├── configs
        │     └── Bilayer_graphene_train.ini   # 训练与推理配置
        ├── data
        │     ├── __init__.py             # 包初始化
        │     ├── data.py                 # 数据集读取与预处理
        │     └── graph.py                # 图数据结构定义（数据预处理侧）
        ├── graph
        │     ├── graph.py                # 图结构与算子实现（模型前向侧）
        │     └── loss.py                 # 图相关损失函数
        └── models
              ├── default_configs         # 默认配置模板
              │     ├── base_default.ini
              │     ├── eval_default.ini
              │     └── train_default.ini
              ├── __init__.py
              ├── e3modules.py            # E(3) 等变模块定义
              ├── kernel.py               # DeepHE3Kernel：训练/评估主流程
              ├── model.py                # 主网络 Net 结构
              ├── parse_configs.py        # 配置解析工具
              └── utils.py                # 训练辅助工具（基函数等）
```

- 模型主体由 `models/model.py` 中的 `Net` 与 `models/e3modules.py` 中的等变模块共同构成；训练流程由 `models/kernel.py` 中的 `DeepHE3Kernel` 封装，包括数据加载、损失计算（如 `L2LossMask`）、学习率调度与日志记录等。

## 模型运行步骤

### 训练

- 确保已完成以下准备：
    - 安装 MindSpore、MindScience 及依赖包；
    - 下载并解压 `Bilayer_graphene_dataset.zip` 至当前目录；
    - 根据需要修改 `configs/Bilayer_graphene_train.ini` 中的训练参数（如批大小、学习率、训练轮数、`checkpoint_dir` 等）。
- 在 `deephe3nn` 目录下执行：

```bash
python train.py configs/Bilayer_graphene_train.ini
```

训练过程中会在配置文件指定的 `checkpoint_dir` 下保存模型权重，同时在日志中输出训练与验证损失。

### 推理

- 将需要加载的权重路径写入配置文件中的 `checkpoint_dir` 字段。
- 在 `deephe3nn` 目录下执行：

```bash
python predict.py configs/Bilayer_graphene_train.ini
```

推理脚本会基于给定的结构配置计算电子哈密顿量，并在日志中输出评估结果，具体输出格式可根据配置与下游任务需求进行调整。

### 训练日志示例

```log
INFO:root:Starting new training process
INFO:root:-------Begin training-------
INFO:root:=================================epoch: 0
...
INFO:root:----------------------eval epoch: 916-------step: 19
INFO:root:evaluating time: 0.25410914421081543
INFO:root:learning rate: 3.159372e-10
INFO:root:val mse loss: 7.4168706e-06
INFO:root:epoch: 916

INFO:root:last train loss: 7.4168706e-06
INFO:root:average eval loss: 6.1306587e-06
INFO:root:Train finished, cost 63180.765609025955 s
INFO:root:best loss: 6.1306587e-06
```

## 许可证

- 开源协议：`Apache License 2.0`
- 许可证链接：`http://www.apache.org/licenses/LICENSE-2.0`

## 引用

- 如果本项目对您的研究有帮助，请引用相关工作：
    - Xiaoxun Gong, He Li, Nianlong Zou, et al. General framework for E(3)-equivariant neural network representation of density functional theory Hamiltonian[J]. Nature Communications, 2023, 14: 2848.
