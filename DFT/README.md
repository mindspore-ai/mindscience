# 密度泛函计算套件

## 概述

密度泛函计算套件基于国产深度学习框架MindSpore开发，提供了一系列量子化学计算的AI解决方案。套件涵盖电荷密度预测、耦合簇能量预测、哈密顿量预测、交换相关势预测等多种应用，适用于材料科学和量子化学研究。

## 应用案例

### 1. ML-DFT计算电荷密度

**路径**: `DFT/ml-dft/`

**功能描述**: ML-DFT是三个深度学习模型的组合，根据输入的POSCAR格式结构信息，在DFT水平上预测分子和聚合物电子结构的各种特性：电子密度、态密度和总势能（包括力和应力张量）。

**主要文件**:
- `ML_DFT.py` - 主脚本
- `config/T1_config.yaml` - 电荷密度预测配置
- `config/T2_config.yaml` - 能量和DOS预测配置
- `src/CHG.py` - 电子密度模型
- `src/DOS.py` - 态密度模型
- `src/Energy.py` - 能量模型
- `src/dataset.py` - 数据处理

**关键特性**:
- 预测电子密度、态密度、总势能
- 支持力和应力张量计算
- POSCAR格式输入

---

### 2. DeepDFT预测电荷密度

**路径**: `DFT/deepdft/`

**功能描述**: Equivariant DeepDFT模型基于等变消息传递神经网络，能够处理旋转等变性的问题。模型通过在图中插入特殊的探测节点来计算密度，仅需要原子序数和原子坐标作为输入，不依赖于预定义的基函数集。

**主要文件**:
- `train.py` - 训练脚本
- `evaluate_model.py` - 评估脚本
- `predict_with_model.py` - 推理脚本
- `configs/config.yaml` - 配置文件
- `src/dataset.py` - 数据处理
- `src/densitymodel.py` - 密度模型
- `src/layer.py` - 网络层

**关键特性**:
- 等变消息传递神经网络
- 不依赖基函数集
- 探测节点计算密度
- 支持QM9、NMC等数据集

---

### 3. Delta-DFT预测耦合簇能量

**路径**: `DFT/delta_dft/`

**功能描述**: 基于机器学习的单分子密度泛函近似耦合簇能量的方法。对分子能量进行归一化，将势能投影到对数-极坐标系做傅里叶变换，以消除旋转对势能场的影响，结合神经网络计算密度泛函结果与耦合簇能量结果的差值后进行对齐。

**主要文件**:
- `train.py` - 训练脚本
- `config.yml` - 配置文件
- `src/dataset.py` - 数据处理
- `src/module.py` - 模型模块
- `src/trainer.py` - 训练器
- `src/utils.py` - 工具函数

**关键特性**:
- 耦合簇能量预测
- 对数-极坐标变换
- 傅里叶变换消除旋转影响
- DFT与CC能量差值学习

---

### 4. DeepH预测哈密顿量

**路径**: `DFT/deephe3nn/`

**功能描述**: DeephE3nn是一个基于E3的等变神经网络，利用晶体中的原子结构去预测体系的电子哈密顿量。

**主要文件**:
- `train.py` - 训练脚本
- `predict.py` - 推理脚本
- `configs/Bilayer_graphene_train_numpy.ini` - 配置文件
- `data/data.py` - 数据集处理
- `data/graph.py` - 图数据结构
- `models/kernel.py` - 主执行流程

**关键特性**:
- E3等变神经网络
- 电子哈密顿量预测
- 支持双层石墨烯数据集

---

### 5. EEDM预测DNA结构电子密度

**路径**: `DFT/e3-dna/`

**功能描述**: EEDM (Equivariant Electron Density Model) 是一个基于E3的等变神经网络，用于预测DNA结构的电子密度。

**主要文件**:
- `train_dna.py` - 训练脚本
- `configs/config.yaml` - 配置文件
- `src/data.py` - 数据集构建
- `src/models.py` - 模型结构
- `src/utils.py` - 辅助函数

**关键特性**:
- E3等变神经网络
- DNA结构电子密度预测
- 适用于生物大分子

---

### 6. ML-DFTXC Potential预测XC势

**路径**: `DFT/ml-dftxc/`

**功能描述**: ML-DFTXC potential model基于三维卷积神经网络，通过映射准局部电子密度到局部交换-相关（XC）势来确定DFT的精确XC势。

**主要文件**:
- `train.py` - 训练脚本
- `xcnn.py` - 推理脚本
- `config/train.cfg` - 训练配置
- `config/test.cfg` - 测试配置
- `src/dataset.py` - 数据集构建
- `src/model.py` - 模型结构
- `src/loss.py` - 损失函数

**关键特性**:
- 三维卷积神经网络
- 交换-相关势预测
- 准局部电子密度映射

---

## 技术特点

1. **等变神经网络**: 多个模型采用E3等变网络，保证物理对称性
2. **多任务预测**: 支持电子密度、能量、哈密顿量等多种性质预测
3. **高效计算**: 相比传统DFT计算大幅提升效率
4. **材料适用**: 支持分子、晶体、DNA等多种体系
5. **端到端学习**: 从原子结构直接预测电子性质

## 数据集与模型权重

### 数据集下载

| 应用 | 数据集下载地址 | 存放路径 |
|------|---------------|---------|
| DeepDFT | https://data.dtu.dk/articles/dataset/QM9_Charge_Densities_and_Energies_Calculated_with_VASP/16794500 (QM9) | `deepdft/data/qm9/` |
| DeepDFT | https://data.dtu.dk/articles/dataset/NMC_Li-ion_Battery_Cathode_Energies_and_Charge_Densities/16837721 (NMC，推荐，约10GB) | `deepdft/data/nmc/` |
| DeepDFT | https://data.dtu.dk/articles/dataset/Ethylene_Carbonate_Molecular_Dynamics/16691825 (Eth) | `deepdft/data/eth/` |
| ML-DFT | https://github.com/Ramprasad-Group/ML-DFT/tree/main/tutorials/database | `ml-dft/dataset/` |
| Delta-DFT | https://github.com/MihailBogojeski/ml-dft/tree/master/water_102 | `delta_dft/Dataset/` |
| DeepH | https://zenodo.org/records/7553640 (Bilayer_graphene_dataset.zip) | `deephe3nn/` |
| E3-DNA | http://aisccc.cn/database/data-details?id=171&type=resource (dataset.zip) | `e3-dna/data/` |
| ML-DFTXC | https://github.com/zhouyyc6782/oep-wy-xcnn/tree/master/example/simple_H2 | `ml-dftxc/data/` |

### 模型权重下载

| 应用 | ckpt下载地址 | 存放路径 |
|------|-------------|---------|
| DeepDFT | https://github.com/12138xs/MindDFT/tree/main/DeepDFT/checkpoints | `deepdft/checkpoints/` |

## 运行环境

- MindSpore >= 2.0
- Python >= 3.7
- NumPy, SciPy
- PySCF（部分应用）

## 使用方式

```bash
# ML-DFT电荷密度预测
cd ml-dft
python ML_DFT.py --config config/T1_config.yaml

# DeepDFT训练
cd deepdft
python train.py --config configs/config.yaml

# Delta-DFT训练
cd delta_dft
python train.py

# DeepH训练
cd deephe3nn
python train.py --config configs/Bilayer_graphene_train_numpy.ini

# EEDM训练
cd e3-dna
python train_dna.py --config configs/config.yaml

# ML-DFTXC训练
cd ml-dftxc
python train.py --config config/train.cfg
```

## 目录结构

```
DFT/
├── README.md
├── ml-dft/           # ML-DFT电荷密度预测
├── deepdft/          # DeepDFT电荷密度预测
├── delta_dft/        # Delta-DFT耦合簇能量
├── deephe3nn/        # DeepH哈密顿量预测
├── e3-dna/           # EEDM DNA电子密度
└── ml-dftxc/         # ML-DFTXC交换相关势
```
