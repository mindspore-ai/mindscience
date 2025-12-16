# PowerFlowNet - MindSpore 实现

MindSpore 版 PowerFlowNet 的完整实现，支持 CPU 和 Ascend NPU 设备。

[English Version](README_EN.md)

## 概述

PowerFlowNet 利用消息传递图神经网络（GNN）进行高质量的电力潮流近似计算。本仓库提供完整的MindSpore 实现，包含 11 种 GNN 架构变体和完整的数据处理管道。

## 项目特点

✅ **完全自实现** - 零外部 GNN 库依赖（MessagePassing、TAGConv、degree 等）  
✅ **Ascend NPU 优化** - 针对华为 Ascend 硬件优化，支持 PYNATIVE_MODE 高效运行  
✅ **11 种模型架构** - MLP、GCN、MPN 及其 7 种变体  
✅ **双数据格式支持** - PowerFlowData（12D）和 PowerFlowDataV2（4D，推荐）  
✅ **经过验证** - 完整的对齐测试和数值稳定性验证  
✅ **Apache 2.0 License** - 基于原始 MIT 版本的合法衍生

## 项目结构

```text
powerflownet/
├── src/                        # 核心源代码
│   ├── __init__.py            # 包导出（MPN、PowerFlowData、PowerFlowDataV2）
│   ├── argument_parser.py     # 参数解析（JSON 配置 + CLI）
│   ├── gnn_ops.py             # GNN 操作（MessagePassing、TAGConv、degree）
│   ├── cpu_npu_ops.py         # CPU/Ascend 兼容层
│   ├── data_utils.py          # 数据工具（Data、DataLoader、InMemoryDataset）
│   ├── power_flow_data.py     # 电力潮流数据处理（5 个类，2 种格式）
│   ├── mpn.py                 # 消息传递网络（9 个 MPN 变体）
│   ├── gcn.py                 # 图卷积网络（GCN、SkipGCN）
│   ├── mlp.py                 # MLP 基线模型
│   ├── training.py            # 训练工具和回调函数
│   ├── evaluation.py          # 评估指标和验证
│   ├── custom_loss_functions.py # 自定义损失函数
│   └── __pycache__/           # Python 缓存
├── configs/                    # 配置文件
│   └── config.py              # 设备配置和 MindSpore 初始化
├── data/                       # 数据目录
│   └── mindspore/             # MindSpore 格式数据（处理后和原始）
├── models/                     # 保存的模型检查点
│   ├── 14/                    # 14 节点系统模型
│   └── 14v2/                  # 14 节点系统 V2 格式模型
├── logs/                       # 训练日志和结果
│   ├── 14/                    # 12D 数据格式训练日志
│   └── 14v2/                  # 4D V2 格式训练日志（推荐）
├── README.md                  # 中文文档
├── README_EN.md               # 英文文档
├── README_MINDSPORE_MIGRATION.md # 迁移详细文档
├── train.py                   # 训练脚本（原始 12D 格式）
├── test.py                    # 评估脚本
├── requirements.txt           # Python 依赖
└── LICENSE                    # Apache 2.0 许可证
```

## 快速开始

### 安装环境

```bash
# 创建 conda 环境
conda create -n mind python=3.9
conda activate mind

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
# MLP 基线模型（快速）
python train.py --model mlp --case 14 --epochs 20

# MPN 消息传递网络（推荐）
python train.py --model mpn --case 14 --epochs 20

# GCN 图卷积网络
python train.py --model gcn --case 14 --epochs 20

# 使用 V2 数据格式（推荐，4D 输入）
# 需要在 train.py 中切换到 PowerFlowDataV2
```

### 评估模型

```bash
# 评估训练好的模型
python test.py --model mlp --run_id <run_id>
```

## 支持的模型

### 基础模型（3 种）

| 模型 | 描述 | 参数量 |
|------|------|--------|
| `mlp` | 多层感知机基线 | 小 |
| `gcn` | 图卷积网络 | 中 |
| `mpn` | 消息传递网络 | 中 |

### MPN 变体（8 种）

| 模型 | 描述 |
|------|------|
| `skip_mpn` | 带跳接连接的 MPN |
| `mask_embed_mpn` | 带掩码嵌入的 MPN |
| `multi_mpn` | 多步消息传递 + 卷积 |
| `mask_embed_multi_mpn` | 掩码嵌入 + 多步 MP |
| `mask_embed_multi_mpn_nomp` | 掩码嵌入 + 多步卷积（无 MP） |
| `mpn_simplenet` | 简化的 MPN |
| `multi_conv_net` | 多平行卷积 |

## 数据格式

### V2 格式（推荐，4D 输入）

最优化和推荐的格式，适合 Ascend NPU：

```text
├── node_features.npy      # (N_samples, N_nodes, 4) - 归一化功率
├── edge_features.npy      # (N_samples, N_edges, 2) - 阻抗
└── edge_index.npy         # (2, N_edges) - 边连接
```

### 原始格式（12D 输入）

数据输入格式：

```text
├── node_features.npy      # (N_samples, N_nodes, 9) - one-hot + 特征
├── edge_features.npy      # (N_samples, N_edges, 7) - 多种边属性
└── edge_index.npy         # (2, N_edges) - 边连接
```

数据集下载：[Surf Drive 链接](https://surfdrive.surf.nl/files/index.php/s/Qw4RHLvI2RPBIBL)

## 关键特性说明

### 1. 完全自实现的 GNN 操作

- **MessagePassing**: 通用图神经网络基类，支持自定义聚合函数
- **TAGConv**: 拓扑自适应图卷积，支持 k-hop 邻域聚合
- **degree 函数**: 计算图节点度数，支持加权度数

### 2. Ascend NPU 优化

- PYNATIVE_MODE + JIT 级别 O0，确保 Ascend 兼容性
- CPU/Ascend 兼容的操作层（gather、scatter、where）
- 无分布式模式强制（RANK_TABLE_FILE 移除）

### 3. 数据处理管道

- **PowerFlowData**: 灵活的多格式数据加载（12D 格式）
- **PowerFlowDataV2**: 优化的向量化数据处理（4D 格式）
- **图批处理**: 支持将多个图合并为单一批次
- **物理约束**: 归一化处理和特征约束

### 4. 完整的训练框架

- 灵活的参数解析（JSON 配置 + CLI）
- 训练回调和早停机制
- 完整的评估指标（MAE、MSE、RMSE 等）

## 环境要求

- **MindSpore**: >= 2.7.0
- **Python**: 3.9.0
- **NumPy**: >= 1.19.0
- **tqdm**: 进度条
- **matplotlib**: 可选，用于可视化

## 许可证说明

本项目采用 Apache 2.0 许可证。代码基于以下原始项目：

**原始项目**：[PowerFlowNet (ericyangyu/PowerFlowNet)](https://github.com/stavrosorf/poweflownet)

- 原始许可证：MIT License
- 迁移内容：框架适配、数据处理、模型架构

**主要改动**：

- MindSpore 框架迁移
- Ascend NPU 针对性优化
- 数据处理管道重构和优化

## 引用

如果您使用本实现，请引用原始论文：

```bibtex
@article{LIN2024110112,
  title = {PowerFlowNet: Power flow approximation using message passing Graph Neural Networks},
  journal = {International Journal of Electrical Power & Energy Systems},
  volume = {160},
  pages = {110112},
  year = {2024},
  issn = {0142-0615},
  doi = {https://doi.org/10.1016/j.ijepes.2024.110112},
  author = {Nan Lin and Stavros Orfanoudakis and Nathan Ordonez Cardenas and Juan S. Giraldo and Pedro P. Vergara},
}
```

## 快速参考

### 导入模型和数据

```python
from src import MPN, PowerFlowDataV2
from src.data_utils import DataLoader

# 加载模型
model = MPN(nfeature_dim=4, efeature_dim=2, output_dim=4,
            hidden_dim=64, n_gnn_layers=3, k=3, dropout_rate=0.1)

# 加载数据
dataset = PowerFlowDataV2(data_path='data/mindspore', case=14)
loader = DataLoader(dataset, batch_size=32)
```

### 训练循环

```python
import mindspore as ms
from mindspore import nn

optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(20):
    for batch in loader:
        def forward_fn(data):
            pred = model(data)
            loss = loss_fn(pred, data.y)
            return loss

        loss, grads = ms.value_and_grad(forward_fn, weights=model.trainable_params())(batch)
        optimizer(grads)
```

## 故障排查

### 问题：Ascend 编译错误

**症状**：`RuntimeError: Can not find kernel tensor for node`  
**解决**：确保在 config.py 中设置了正确的设备模式：

```python
ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, jit_config=ms.JitConfig(jit_level="O0"))
```

### 问题：内存溢出

**症状**：OOM 错误  
**解决**：减小批次大小或使用 PowerFlowDataV2（更高效的内存使用）

### 问题：数据加载失败

**症状**：文件未找到  
**解决**：确保数据文件在 `data/mindspore/processed/` 目录中

## 文档资源

- [src/argument_parser.py](src/argument_parser.py) - 参数解析文档。
- [src/mpn.py](src/mpn.py) - MPN 架构说明。
- [src/power_flow_data.py](src/power_flow_data.py) - 数据处理详解。
