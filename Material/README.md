# 材料结构预测套件

## 概述

材料结构预测套件基于国产深度学习框架MindSpore开发，提供了一系列材料性质预测和结构生成的AI解决方案。套件涵盖分子势能预测、晶体结构生成、材料性质预测等多种应用，适用于材料科学研究和新材料设计。

## 应用案例

### 1. Allegro

**路径**: `Material/applications/allegro/`

**功能描述**: Allegro是基于等变图神经网络构建的SOTA模型，可以在大规模材料体系中进行高精度预测。该模型相关论文已发表在Nature Communications期刊上，主要用于分子势能预测，具有较高的应用价值。

**主要文件**:
- `train.py` - 训练脚本
- `predict.py` - 推理脚本
- `rmd.yaml` - 配置文件
- `src/allegro_embedding.py` - embedding模块
- `src/potential.py` - 势能网络模块
- `src/trainer.py` - 训练脚本
- `src/predicter.py` - 推理评估脚本

**数据集**: rMD17数据集

**关键特性**:
- 等变图神经网络
- 大规模体系预测
- 高精度势能预测

---

### 2. CDVAE (Crystal Diffusion Variational AutoEncoder)

**路径**: `Material/applications/cdvae/`

**功能描述**: CDVAE是用来生成材料周期性结构的SOTA模型，相关论文已发表在ICLR上。模型包含encoder和decoder两部分：encoder将输入信息转化成隐变量z；decoder部分通过MLP和扩散模型生成原子种类、原子位置等结构信息。

**主要文件**:
- `train.py` - 训练脚本
- `evaluation.py` - 推理脚本
- `compute_metrics.py` - 评估脚本
- `conf/` - 配置文件目录
- `src/dataloader.py` - 数据加载
- `src/evaluate_utils.py` - 推理结果生成
- `src/metrics_utils.py` - 评估结果计算

**数据集**: Perov_5、Carbon_24、MP_20

**关键特性**:
- 扩散模型+变分自编码器
- 周期性结构生成
- 新材料设计

---

### 3. DeephE3nn

**路径**: `Material/applications/deephe3nn/`

**功能描述**: DeephE3nn是一个基于E3的等变神经网络，利用晶体中的原子结构去预测体系的电子哈密顿量。这是一个专门用于量子化学计算的应用。

**主要文件**:
- `train.py` - 训练脚本
- `predict.py` - 推理脚本
- `configs/` - 配置文件目录
- `data/data.py` - 数据集处理
- `data/graph.py` - 图数据结构
- `models/kernel.py` - 主执行流程

**数据集**: Bilayer_graphene数据集

**关键特性**:
- E3等变神经网络
- 电子哈密顿量预测
- 量子化学计算

---

### 4. Matformer

**路径**: `Material/applications/matformer/`

**功能描述**: Matformer是基于图神经网络和Transformer架构的SOTA模型，用于预测晶体材料的各种性质。结合了图神经网络对晶体结构的建模能力和Transformer的注意力机制。

**主要文件**:
- `train.py` - 训练脚本
- `predict.py` - 推理脚本
- `config.yaml` - 配置文件
- `data/data.py` - 数据集处理
- `data/features.py` - 特性处理
- `data/generate.py` - 图数据生成
- `data/graphs.py` - 图数据结构

**数据集**: jdft_3d数据集

**关键特性**:
- 图神经网络+Transformer
- 晶体材料性质预测
- 注意力机制

---

### 5. NequIP

**路径**: `Material/applications/nequip/`

**功能描述**: NequIP是基于等变图神经网络构建的SOTA模型，相关论文已发表在Nature Communications期刊上。该模型验证了在分子势能与力的预测中的有效性，具有较高的应用价值。

**主要文件**:
- `train.py` - 训练脚本
- `predict.py` - 推理脚本
- `rmd.yaml` - 配置文件
- `src/dataset.py` - 数据集处理
- `src/trainer.py` - 训练脚本
- `src/predicter.py` - 推理评估脚本
- `src/plot.py` - 结果作图

**数据集**: rMD17数据集

**关键特性**:
- 等变图神经网络
- 分子势能与力预测
- 分子动力学模拟

---

### 6. ORB

**路径**: `Material/applications/orb/`

**功能描述**: ORB是一个基于图神经网络（GNN）的机器学习力场（MLFF）模型，设计为通用的原子间势能模型，适用于多种模拟任务（几何优化、蒙特卡洛模拟和分子动力学模拟）。在Matbench Discovery基准测试中，ORB模型的误差比其他方法降低了31%，并且在大系统规模下的速度比MACE提高了3-6倍。

**主要文件**:
- `finetune.py` - 微调脚本
- `evaluate.py` - 评估脚本
- `configs/config.yaml` - 单卡训练配置
- `configs/config_parallel.yaml` - 多卡并行训练配置
- `configs/config_eval.yaml` - 推理配置
- `src/ase_dataset.py` - 数据集处理和加载
- `src/atomic_system.py` - 原子系统数据结构
- `src/pretrained.py` - 预训练模型相关函数
- `src/trainer.py` - 模型loss类定义

**数据集**: mptrj数据集

**关键特性**:
- 通用机器学习力场
- 几何优化、MD模拟
- 大规模并行训练
- 比MACE快3-6倍

---

## 技术特点

1. **等变神经网络**: 多个模型采用E3等变网络，保证物理对称性
2. **SOTA性能**: 多个模型发表于Nature Communications、ICLR等顶级期刊
3. **结构生成**: 支持新材料结构生成和设计
4. **高效预测**: 相比传统方法大幅提升效率
5. **并行训练**: 支持大规模分布式训练

## 数据集与模型权重

数据集及ckpt下载地址：https://download-mindspore.osinfra.cn/mindscience/mindchemistry/

## 运行环境

- MindSpore >= 2.0
- Python >= 3.7
- NumPy, SciPy, Matplotlib
- ASE（部分应用）

## 使用方式

```bash
# Allegro训练
cd applications/allegro
python train.py --config rmd.yaml

# CDVAE训练
cd applications/cdvae
python train.py

# DeephE3nn训练
cd applications/deephe3nn
python train.py

# Matformer训练
cd applications/matformer
python train.py --config config.yaml

# NequIP训练
cd applications/nequip
python train.py --config rmd.yaml

# ORB微调
cd applications/orb
python finetune.py --config configs/config.yaml
```

## 目录结构

```
Material/
├── README.md
└── applications/
    ├── allegro/      # Allegro势能预测
    ├── cdvae/        # CDVAE结构生成
    ├── deephe3nn/    # DeephE3nn哈密顿量
    ├── matformer/    # Matformer性质预测
    ├── nequip/       # NequIP势能预测
    └── orb/          # ORB机器学习力场
```
