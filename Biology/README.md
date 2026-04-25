# 生物计算套件

## 概述

生物计算套件基于国产深度学习框架MindSpore开发，提供了一系列蛋白质和RNA结构预测与设计的AI解决方案。套件涵盖蛋白质功能预测、蛋白质语言模型、反向折叠、序列设计、复合物结构预测和RNA二级结构预测等多种应用。

## 应用案例

### 1. DeepFRI蛋白质功能预测

**路径**: `MindSPONGE/applications/model_configs/DeepFri/`

**功能描述**: DeepFRI是一种图形卷积网络，通过利用从蛋白质语言模型和蛋白质结构中提取的序列特征来预测蛋白质功能。可对蛋白质进行四个方面的预测：分子功能(MF)、细胞组分(CC)、生物过程(BP)和EC编号。

**主要文件**:
- `model_configs/DeepFri/deepfri_bp.yaml` - 生物过程配置
- `model_configs/DeepFri/deepfri_cc.yaml` - 细胞组分配置
- `model_configs/DeepFri/deepfri_mf.yaml` - 分子功能配置
- `model_cards/DeepFri.md` - 模型说明文档

**关键特性**:
- 基于图卷积网络
- 结合蛋白质语言模型特征
- 支持GO术语和EC编号预测
- 多标签分类任务

---

### 2. ESM-2蛋白质语言模型

**路径**: `MindSPONGE/applications/model_configs/ESM2/`

**功能描述**: ESM-2是迄今为止训练的最大的蛋白质语言模型。基于transformer的语言模型，使用注意力机制学习输入序列中氨基酸对之间的相互作用。可用于蛋白质序列表示学习和结构预测。比前代模型ESM-1b有实质性改进，150M参数模型性能优于650M参数的ESM-1b。

**主要文件**:
- `model_configs/ESM2/esm2_config.yaml` - 模型配置
- `model_cards/ESM-2.md` - 模型说明文档

**关键特性**:
- 基于Transformer架构
- 大规模蛋白质序列预训练
- 支持序列表示提取
- 可用于下游任务微调

---

### 3. ESM-IF1反向折叠模型

**路径**: `MindSPONGE/applications/model_configs/ESM_IF1/`

**功能描述**: 反向折叠模型，通过蛋白质骨架的原子坐标预测蛋白质序列。用于从头蛋白质设计。使用Geometric Vector Perceptron(GVP)层学习向量特征的等变转换。训练数据来自AlphaFold2预测的1200万蛋白质结构。

**主要文件**:
- `model_configs/ESM_IF1/sampling.yaml` - 推理配置
- `model_configs/ESM_IF1/training.yaml` - 训练配置
- `model_cards/ESM-IF1.md` - 模型说明文档

**关键特性**:
- 基于GVP的等变网络
- 从骨架结构预测序列
- 支持训练和推理
- 适用于蛋白质设计

---

### 4. ProteinMPNN蛋白质序列设计

**路径**: `MindSPONGE/applications/model_configs/Proteinmpnn/`

**功能描述**: 基于深度学习的蛋白质序列设计方法。给定蛋白质的backbone结构，预测能折叠成该结构的氨基酸序列。支持单链或多链之间的氨基酸序列耦合，广泛适用于单体、环状低聚物、蛋白质纳米颗粒等设计。引入随机位点解码机制和主链高斯噪音增强泛化能力。

**主要文件**:
- `model_configs/Proteinmpnn/proteinmpnn_predict.yaml` - 推理配置
- `model_configs/Proteinmpnn/proteinmpnn_train.yaml` - 训练配置
- `model_cards/ProteinMPNN.MD` - 模型说明文档

**关键特性**:
- 基于图神经网络
- 支持多链蛋白质设计
- 随机位点解码
- 高斯噪音增强

---

### 5. Multimer蛋白质复合物结构预测

**路径**: `MindSPONGE/applications/model_configs/Multimer/`

**功能描述**: AlphaFold Multimer是蛋白质复合物结构预测模型。在AlphaFold 2基础上针对复合物结合界面结构做了调整，支持多链特征提取和对称置换。适用于预测蛋白质复合物的三维结构，修改了损失函数以处理同源多聚物的对称置换问题。

**主要文件**:
- `model_configs/Multimer/predict_256.yaml` - 256分辨率配置
- `model_configs/Multimer/predict_512.yaml` - 512分辨率配置
- `model_configs/Multimer/predict_768.yaml` - 768分辨率配置
- `model_configs/Multimer/predict_1024.yaml` - 1024分辨率配置
- `model_configs/Multimer/predict_1280.yaml` - 1280分辨率配置
- `model_configs/Multimer/predict_1536.yaml` - 1536分辨率配置
- `model_configs/Multimer/predict_1792.yaml` - 1792分辨率配置
- `model_cards/afmultimer.md` - 模型说明文档

**关键特性**:
- 基于AlphaFold 2架构
- 支持多链蛋白质复合物
- 多种分辨率配置
- 处理对称置换问题

---

### 6. UFold RNA二级结构预测

**路径**: `MindSPONGE/applications/model_configs/UFold/`

**功能描述**: 基于深度学习的RNA二级结构预测方法。直接根据注释数据和碱基配对规则进行训练，使用完全卷积网络进行预测。输入为核苷酸序列转换的17通道L×L矩阵，输出为RNA二级结构的L×L接触矩阵。比传统热力学模型速度更快、准确度更高。

**主要文件**:
- `model_configs/UFold/ufold_config.yaml` - 模型配置
- `model_cards/UFold.md` - 模型说明文档

**关键特性**:
- 基于完全卷积网络
- 端到端预测
- 17通道输入表示
- 高准确度预测

---

## 技术特点

1. **功能全面**: 涵盖蛋白质功能预测、结构预测、序列设计全流程
2. **模型先进**: 采用ESM-2、AlphaFold Multimer等SOTA模型
3. **支持训练**: 部分模型支持本地训练和微调
4. **多模态**: 支持蛋白质和RNA两类生物大分子
5. **工业级**: 支持实际蛋白质设计应用

## 运行环境

- MindSpore >= 2.0
- MindSPONGE
- Python >= 3.7

## 使用方式

通过MindSPONGE应用模块加载配置运行：

```python
from mindsponge import PipeLine

# 加载模型配置
pipeline = PipeLine(name='DeepFri')
pipeline.model.from_config('model_configs/DeepFri/deepfri_mf.yaml')

# 运行预测
result = pipeline.predict(protein_sequence)
```

## 目录结构

```
Biology/MindSPONGE/
├── applications/
│   ├── model_configs/       # 模型配置文件
│   │   ├── DeepFri/         # DeepFRI配置
│   │   ├── ESM2/            # ESM-2配置
│   │   ├── ESM_IF1/         # ESM-IF1配置
│   │   ├── Proteinmpnn/     # ProteinMPNN配置
│   │   ├── Multimer/        # Multimer配置
│   │   └── UFold/           # UFold配置
│   ├── model_cards/         # 模型说明文档
│   └── research/            # 研究项目
├── src/                     # 源代码
│   ├── mindsponge/          # MindSPONGE核心
│   └── sponge/              # SPONGE模块
└── tutorials/               # 教程文档
```
