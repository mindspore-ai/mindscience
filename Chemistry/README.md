# 化学反应设计和高效模拟套件

## 概述

化学反应设计和高效模拟套件基于国产深度学习框架MindSpore开发，提供了一系列材料性质预测和晶体结构生成的AI解决方案。套件涵盖材料关键性质预测、晶体结构生成等多种应用，适用于新材料设计和化学反应研究。

## 应用案例

### 1. BETE-NET-Lamb测试

**路径**: `Chemistry/applications/bete-net-lamb/`

**功能描述**: BETE-NET模型训练和推理系统，用于预测材料的三个关键性质：Formation energy (λ)、Phonon frequency (ω_log)、Density (ω_2)。基于MindSpore框架实现的深度学习模型，支持完整的训练流程和推理评估。

**主要文件**:
- `README.md` - 说明文档
- `Pred_FPD_ms.py` - 推理脚本
- `run_full_training.py` - 完整训练脚本
- `start_training_with_batch.py` - 交互式训练启动器
- `indices/` - 训练索引文件
- `structures/` - 结构数据

**关键特性**:
- 训练指标：MAE、RMSE、R²
- 支持早停机制和训练状态保存
- 实时进度条和可视化输出
- Lamb优化器

---

### 2. BETE-NET-W2测试

**路径**: `Chemistry/applications/bete-net-w2/`

**功能描述**: BETE-NET模型的w2变体，与bete-net-lamb功能相似，预测材料的三个关键性质。采用不同的训练策略或模型配置。

**主要文件**:
- `README.md` - 说明文档
- `Pred_FPD_ms.py` - 推理脚本
- `run_full_training.py` - 完整训练脚本
- `start_training_with_batch.py` - 交互式训练启动器
- `indices/` - 训练索引文件
- `structures/` - 结构数据

**关键特性**:
- W2优化策略
- 材料性质预测
- 完整训练流程

---

### 3. BETE-NET-Wlog测试

**路径**: `Chemistry/applications/bete-net-wlog/`

**功能描述**: BETE-NET模型的wlog变体，同样用于材料性质预测。可能采用不同的损失函数或数据处理方式。

**主要文件**:
- `README.md` - 说明文档
- `Pred_FPD_ms.py` - 推理脚本
- `run_full_training.py` - 完整训练脚本
- `start_training_with_batch.py` - 交互式训练启动器
- `indices/` - 训练索引文件
- `structures/` - 结构数据

**关键特性**:
- Wlog损失函数
- 材料性质预测
- 数据处理优化

---

### 4. CrystalFlow测试

**路径**: `Chemistry/applications/crystalflow/`

**功能描述**: 晶体结构生成模型，基于神经常微分方程和归一化流模型。给定组分，预测晶体材料的结构。相比扩散模型，具有更简洁、灵活、高效的优点。在MP20等基准数据集上达到优秀水平。

**主要文件**:
- `README.md` - 说明文档
- `train.py` - 训练脚本
- `evaluate.py` - 推理脚本
- `compute_metric.py` - 评估脚本
- `config.yaml` - 配置文件
- `models/flow.py` - 流模型模块
- `models/cspnet.py` - 基于图神经网络的去噪器
- `data/dataset.py` - 数据集处理

**支持数据集**:
- perov_5（钙钛矿）
- carbon_24（碳晶体）
- mp_20（晶胞内原子数≤20）
- mpts_52（晶胞内原子数≤52）

**关键特性**:
- 神经常微分方程
- 归一化流模型
- 晶体结构生成
- 比扩散模型更高效

---

### 5. DiffCSP Carbon_24测试

**路径**: `Chemistry/applications/diffcsp/`

**功能描述**: 基于图神经网络和等变扩散模型的晶体生成模型。给定组分，预测晶体材料的结构。Carbon_24测试使用碳晶体数据集进行训练和评估。

**主要文件**:
- `README.md` - 说明文档
- `train.py` - 训练脚本
- `evaluate.py` - 推理脚本
- `compute_metric.py` - 评估脚本
- `config.yaml` - mp-20配置
- `config_carbon24.yaml` - carbon_24配置
- `models/diffusion.py` - 扩散模型模块
- `models/cspnet.py` - 基于图神经网络的去噪器

**评估指标**:
- match_rate（匹配率）
- rms_dist（均方根距离）

**关键特性**:
- 等变扩散模型
- 碳晶体结构生成
- 图神经网络去噪器

---

### 6. DiffCSP MP-20测试

**路径**: `Chemistry/applications/diffcsp/`

**功能描述**: DiffCSP模型在MP-20数据集上的测试。MP-20数据集包含晶胞内原子数最多为20的晶体结构。

**主要文件**:
- `config.yaml` - mp-20配置文件
- 其他文件同Carbon_24测试

**关键特性**:
- MP-20数据集
- 晶体结构生成
- 多样化材料设计

---

## 技术特点

1. **BETE-NET系列**（3个变体）：专注于材料性质预测，预测Formation energy、Phonon frequency和Density
2. **CrystalFlow**：基于流模型的晶体结构生成，比扩散模型更高效
3. **DiffCSP**：基于扩散模型的晶体结构生成，支持多个数据集配置
4. **MindSpore框架**：所有应用均基于MindSpore框架实现
5. **完整流程**：每个应用都有完整的训练、推理、评估流程

## 数据集与模型权重

数据集及ckpt下载地址：https://download-mindspore.osinfra.cn/mindscience/mindchemistry/

## 运行环境

- MindSpore >= 2.0
- MindChemistry
- Python >= 3.7
- NumPy, SciPy, Matplotlib

## 使用方式

```bash
# BETE-NET-Lamb训练
cd Chemistry/applications/bete-net-lamb
python run_full_training.py

# BETE-NET-Lamb推理
python Pred_FPD_ms.py

# CrystalFlow训练
cd Chemistry/applications/crystalflow
python train.py --config config.yaml

# CrystalFlow评估
python evaluate.py
python compute_metric.py

# DiffCSP Carbon_24训练
cd Chemistry/applications/diffcsp
python train.py --config config_carbon24.yaml

# DiffCSP MP-20训练
python train.py --config config.yaml
```

## 目录结构

```
Chemistry/
├── README.md
├── download_datasets.sh
├── applications/
│   ├── bete-net-lamb/    # BETE-NET Lamb变体
│   ├── bete-net-w2/      # BETE-NET W2变体
│   ├── bete-net-wlog/    # BETE-NET Wlog变体
│   ├── crystalflow/      # CrystalFlow晶体生成
│   └── diffcsp/          # DiffCSP晶体生成
└── mindchemistry/        # 核心模块
    ├── cell/             # 网络单元
    ├── e3/               # E3等变模块
    ├── graph/            # 图神经网络
    └── utils/            # 工具函数
```
