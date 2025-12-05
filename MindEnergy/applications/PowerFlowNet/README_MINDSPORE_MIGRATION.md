# MindSpore PowerFlowNet 迁移文档

## 概述

本文档说明 PowerFlowNet 从 PyTorch 到 MindSpore 的迁移进度，以及如何使用 MindSpore 版本进行训练和测试。

## 迁移状态

### ✅ 已完成迁移

- **src/** - 所有神经网络模型（MLP、MPN、GCN 等 9 个模型）
- **gnn_ops.py** - 自定义 GNN 操作（TAGConv、GCNConv 等）
- **custom_loss_functions.py** - 3 种自定义损失函数
- **power_flow_data.py** - 完整数据加载系统（遗留格式和 V2 格式）
- **data_utils.py** - 数据结构和工具函数
- **cpu_npu_ops.py** - CPU/NPU/Ascend 兼容操作集
- **configs/config.py** - 设备配置和初始化

### 📝 本 PR 包含

- ✅ `src/` 文件夹：11 个 Python 文件，所有模型和工具
- ✅ `configs/` 文件夹：1 个 Python 文件，设备管理和配置
- ✅ 每个文件都包含 Apache License 2.0 copyright 头
- ✅ 完整的数据加载和处理管道
- ✅ Ascend 设备优化

### ⚠️ 本 PR 不包含

- 训练脚本（train.py）
- 测试脚本
- 数据文件和预训练模型
- 基准测试代码

## 项目结构

```text
powerflownet/
├── src/
│   ├── __init__.py                    # 模块导出和公共接口
│   ├── mlp.py                         # MLP 模型实现
│   ├── mpn.py                         # MPN、SkipMPN、MaskEmbdMPN 等多个 GNN 模型
│   ├── gcn.py                         # Graph Convolutional Network
│   ├── gnn_ops.py                     # GNN 基础操作 (MessagePassing, TAGConv, GCNConv)
│   ├── custom_loss_functions.py       # 自定义损失函数 (MaskedL2Loss, PowerImbalance, etc.)
│   ├── power_flow_data.py             # 数据加载类 (PowerFlowData, PowerFlowDataV2 等)
│   ├── data_utils.py                  # 数据工具函数和数据结构
│   ├── cpu_npu_ops.py                 # CPU/NPU/Ascend 兼容操作
│   ├── argument_parser.py             # 参数解析工具
│   ├── training.py                    # 训练工具函数（可选）
│   └── evaluation.py                  # 评估工具函数（可选）
├── configs/
│   └── config.py                      # 设备配置和初始化 (DeviceConfig 类)
└── README_MINDSPORE_MIGRATION.md      # 本文档
```

### 核心模块说明

| 模块 | 功能 | 关键内容 |
|------|------|---------|
| **mlp.py** | MLP 模型 | MLPNet 类 |
| **mpn.py** | 消息传递网络族 | MPN, SkipMPN, MaskEmbdMPN, MultiMPN, MaskEmbdMultiMPN, MaskEmbdMultiMPNNoMP, MultiConvNet, MPNSimplenet, WrappedMultiConv |
| **gcn.py** | 图卷积网络 | GCNNet 类 |
| **gnn_ops.py** | GNN 基础操作 | MessagePassing, TAGConv, GCNConv, degree 等 |
| **custom_loss_functions.py** | 损失函数 | MaskedL2Loss, PowerImbalance, MixedMSEPowerImbalance |
| **power_flow_data.py** | 数据加载 | PowerFlowData, PowerFlowDataLoader, PowerFlowDataV2, PowerFlowDataLoaderV2 |
| **data_utils.py** | 数据工具 | Data, InMemoryDataset, Graph, DataLoader 等数据结构 |
| **cpu_npu_ops.py** | Ascend 优化操作 | gather, pow, where, degree, randint 等在 Ascend 上的优化实现 |
| **config.py** | 设备配置 | DeviceConfig 类，支持 CPU/GPU/Ascend 设备初始化 |

## MindSpore 版本兼容性

- **推荐**: MindSpore 2.7.0+
- **设备支持**:
    - Ascend (华为云芯片)

## Ascend 设备特殊处理

本迁移包含针对 Ascend 设备的重要优化：

### 环境变量处理

```python
# 训练脚本需要在导入之前添加以下代码
if 'RANK_TABLE_FILE' in os.environ:
    del os.environ['RANK_TABLE_FILE']
```

**原因**: `RANK_TABLE_FILE` 会强制 Ascend 启用分布式训练模式，导致 JIT level 被设置为 O2，造成优化器编译问题。

### 上下文配置

```python
import mindspore as ms

# 使用 PYNATIVE_MODE（推荐）
ms.set_context(mode=ms.PYNATIVE_MODE)

# 对于 Ascend，禁用 JIT 优化
if device == 'Ascend':
    ms.set_context(jit_config={"jit_level": "O0"})
```

## 自实现的 torch_geometric 模块

本 PR 不依赖 torch_geometric，而是完整实现了核心的图神经网络模块（在 MindSpore 框架中）。以下是实现的内容对照表：

### data_utils.py - 数据处理模块

| torch_geometric | 本实现 | 说明 |
|---|---|---|
| `torch_geometric.data.Data` | `Data` 类 | 图数据容器，支持任意属性 |
| `torch_geometric.data.InMemoryDataset` | `InMemoryDataset` 类 | 内存数据集基类，支持预处理和缓存 |
| `torch_geometric.data.Graph` | `Graph` 类 | 图数据结构，包含节点/边特征 |
| `torch_geometric.loader.DataLoader` | `DataLoader` 类 | 数据加载器，支持批处理和混洗 |
| `torch_geometric.utils.train_test_split` | `create_data_splits()` | 数据集划分为训练/验证/测试集 |

**关键特性**：

- ✅ 完全独立的实现，无外部依赖
- ✅ 支持 MindSpore Tensor 和 NumPy 数组
- ✅ 动态属性管理
- ✅ 内存高效的批处理

### gnn_ops.py - GNN 操作模块

| torch_geometric | 本实现 | 说明 |
|---|---|---|
| `torch_geometric.nn.MessagePassing` | `MessagePassing` 类 | 消息传递基类，支持 add/mean/max 聚集 |
| `torch_geometric.nn.TAGConv` | `TAGConv` 类 | 拓扑自适应图卷积 |
| `torch_geometric.nn.GCNConv` | `GCNConv` 类 | 图卷积网络层 |
| `torch_geometric.utils.degree` | `degree()` 函数 | 计算节点度数 |
| `torch_geometric.utils.to_undirected` | `to_undirected()` 函数 | 转换为无向图 |

**MessagePassing 特性**：

- ✅ 聚集方法：add, mean, max
- ✅ 消息传递流程：message() → aggregate() → update()
- ✅ 自动参数过滤（使用 inspect 模块）
- ✅ CPU/NPU 兼容的 segment 操作

**TAGConv 实现亮点**：

- ✅ k-hop 拓扑适应性聚集
- ✅ 对称归一化 (D^-0.5 A D^-0.5)
- ✅ 完全匹配原论文和 torch_geometric 实现
- ✅ 数值对齐测试通过

**GCNConv 特性**：

- ✅ 标准图卷积
- ✅ 对称归一化
- ✅ 自环处理

**兼容操作**：

- ✅ 所有操作都使用 MindSpore mint 和 ops 实现
- ✅ 针对 Ascend 设备优化
- ✅ 避免索引操作，使用矩阵乘法替代

## 关键技术点

### 1. TAGConv 算子

- MPN 中使用的 TAGConv (Topology Adaptive GNN Convolution)
- 自定义实现，支持 CPU/Ascend
- 数值对齐测试完成

### 2. Ascend 优化操作

提供了以下操作的 Ascend 优化实现（在 `cpu_npu_ops.py`）：

- `gather_cpu_npu_compatible()` - 张量聚集操作
- `pow_cpu_npu_compatible()` - 张量幂次运算
- `where_cpu_npu_compatible()` - 条件选择
- `degree_cpu_npu_compatible()` - 图度数计算
- `randint_like_cpu_npu_compatible()` - 随机整数生成

### 3. 损失函数

- **MaskedL2Loss** - 基础掩码 L2 损失，用于关键节点预测
- **PowerImbalance** - 物理信息损失，基于功率不平衡
- **MixedMSEPowerImbalance** - 混合损失，综合 MSE 和功率不平衡

### 4. 数据处理

支持两种数据集格式的无缝切换：

- **PowerFlowDataV2** - 100K 样本，4D 特征，推荐使用
- **PowerFlowData** - 较小数据集，12D 特征，向后兼容

## 数据集格式

支持两种数据集格式：

### V2 格式（推荐）

- 100K 样本
- 4D 节点特征：[voltage_mag, voltage_angle, Pd, Qd]
- 使用 `PowerFlowDataV2` 和 `PowerFlowDataLoaderV2`
- case 名称后缀: `v2`（如 `14v2`, `118v2`）

### legacy格式

- 较小数据集
- 12D 节点特征
- 使用 `PowerFlowData` 和 `PowerFlowDataLoader`
- case 名称无后缀（如 `14`, `118`）

## 代码质量和兼容性

### ✅ 代码规范

- 所有文件都有 Apache License 2.0 copyright 头
- 遵循 Python PEP 8 风格指南
- 完整的文档字符串和类型注解

### ✅ 数值验证

- 所有模型在 CPU和Ascend 上数值一致
- 梯度计算通过对数检验（Gradient Checker）
- TAGConv 操作完整对齐测试

### ✅ 设备兼容性

- **Ascend** - 华为云完全支持（含特殊优化）

### ✅ MindSpore 版本

- 最低版本：2.0.0
- 推荐版本：2.7.0+
- 向后兼容性：支持大部分 2.x 版本

## 已知问题和解决方案

### Ascend 首个 epoch 耗时长

**问题**: 第一个 epoch 可能需要 15-30 秒
**原因**: Ascend 需要编译计算图
**解决**: 这是正常行为，后续 epoch 会快速完成

### PYTHONPATH 警告

```text
Can not find the tbe operator implementation
```

**问题**: TBE (Tensor Boost Engine) 路径配置
**影响**: 无，MindSpore 使用其他算子实现
**处理**: 可以安全忽略

## 快速开始

### 安装依赖

```bash
# MindSpore (CPU/GPU)
pip install mindspore

# MindSpore Ascend (在华为云环境)
# 按照官方文档安装：https://www.mindspore.cn/install

# 其他依赖
pip install numpy torch-geometric h5py matplotlib
```

### 导入使用

```python
from powerflownet.src import MLPNet, MPN, GCNNet
from powerflownet.src import PowerFlowDataV2, PowerFlowDataLoaderV2
from powerflownet.configs import init_device

# 初始化设备
init_device('Ascend')  # 或 'CPU', 'GPU'

# 创建模型
model = MPN(
    nfeature_dim=4,
    efeature_dim=2,
    output_dim=4,
    hidden_dim=64,
    n_gnn_layers=3,
    k=3
)

# 加载数据
dataset = PowerFlowDataV2(root='./data', case='14v2', split=[0.7, 0.15, 0.15], task='train')
dataloader = PowerFlowDataLoaderV2(dataset, batch_size=256, shuffle=True)

# 训练
import mindspore as ms
from mindspore import nn

optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)
loss_fn = MaskedL2Loss(regularize=False)

def forward_fn(batch):
    pred = model(batch)
    loss = loss_fn(pred, batch.y, batch.pred_mask)
    return loss

grad_fn = ms.value_and_grad(forward_fn, None, model.trainable_params())

for batch in dataloader:
    loss, grads = grad_fn(batch)
    optimizer(grads)
```

### 完整训练示例

详见项目中的 `train.py` 脚本（不包含在本 PR 中）。

## 支持的模型列表

本 PR 包含以下 **9 个 GNN 模型**的完整 MindSpore 实现：

### 基础模型

1. **MLPNet** - 多层感知机，用作基准模型
2. **GCNNet** - 图卷积网络

### 消息传递网络族（MPN）

3. **MPN** - 基础消息传递网络（带 TAGConv）
4. **SkipMPN** - 带跳跃连接的 MPN
5. **MaskEmbdMPN** - 带掩码嵌入的 MPN
6. **MultiMPN** - 多头 MPN
7. **MaskEmbdMultiMPN** - 带掩码嵌入的多头 MPN
8. **MaskEmbdMultiMPNNoMP** - 无消息传递的多头 MPN
9. **MPNSimplenet** - 简化版 MPN

### 高级模型

10. **MultiConvNet** - 多卷积网络
11. **WrappedMultiConv** - 包装的多卷积网络

### 推荐使用

- **数据格式：4D 特征** → mlp, mpn, gcn, mask_embd_multi_mpn, mpn_simplenet
- **数据格式：12D 特征** → skip_mpn, mask_embd_mpn, multi_mpn, mask_embd_multi_mpn_nomp, multi_conv_net

## API 参考

### 创建模型

```python
# MLP
from powerflownet.src import MLPNet
model = MLPNet(nfeature_dim=4, output_dim=4, hidden_dim=64, n_layers=3)

# MPN 系列
from powerflownet.src import MPN, SkipMPN, MaskEmbdMultiMPN
model = MPN(nfeature_dim=4, efeature_dim=2, output_dim=4, hidden_dim=64, n_gnn_layers=3, k=3)
model = SkipMPN(nfeature_dim=4, efeature_dim=2, output_dim=4, hidden_dim=64, n_gnn_layers=3, k=3)
model = MaskEmbdMultiMPN(nfeature_dim=4, efeature_dim=2, output_dim=4, hidden_dim=64, n_gnn_layers=3, k=3)

# GCN
from powerflownet.src import GCNNet
model = GCNNet(nfeature_dim=4, output_dim=4, hidden_dim=64, n_gnn_layers=3)
```

### 数据加载

```python
# V2 格式（推荐）
from powerflownet.src import PowerFlowDataV2, PowerFlowDataLoaderV2
dataset = PowerFlowDataV2(root='./data', case='14v2', split=[0.7, 0.15, 0.15], task='train')
loader = PowerFlowDataLoaderV2(dataset, batch_size=256, shuffle=True)

# 遗留格式
from powerflownet.src import PowerFlowData, PowerFlowDataLoader
dataset = PowerFlowData(root='./data', case='14', split=[0.7, 0.15, 0.15], task='train')
loader = PowerFlowDataLoader(dataset, batch_size=256, shuffle=True)
```

### 设备初始化

```python
from powerflownet.configs import init_device, DeviceConfig

# 初始化设备
init_device('Ascend')  # 'CPU', 'GPU', 'Ascend'

# 获取设备配置
config = DeviceConfig()
device = config.get_device_target()
```

## 常见问题

### Q: 如何在 Ascend 上运行？

A: 需要在训练脚本最开始添加：

```python
if 'RANK_TABLE_FILE' in os.environ:
    del os.environ['RANK_TABLE_FILE']
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)
```

### Q: 4D 和 12D 特征有什么区别？

A: 4D 特征（推荐）：[voltage_mag, voltage_angle, Pd, Qd]
12D 特征（遗留）：[one-hot(4) + features(4) + mask(4)]

### Q: 是否支持分布式训练？

A: 本 PR 包含的代码不支持分布式训练。如需分布式，需要额外配置。

### Q: 如何扩展新的损失函数？

A: 在 `custom_loss_functions.py` 中继承 `nn.Cell`，实现 `construct()` 方法。

## 贡献指南

本 PR 专注于模型和配置的核心迁移。如需：

- 添加新模型 → 参考 `mpn.py` 的编写风格
- 新损失函数 → 参考 `custom_loss_functions.py`
- 优化 Ascend 性能 → 参考 `cpu_npu_ops.py`

请确保：

1. 添加 Apache License 2.0 copyright 头
2. 包含完整的文档字符串
3. 在三个平台（CPU/GPU/Ascend）上测试数值

## 许可证

Apache License 2.0 - 详见每个源文件的 copyright 头部

## 相关资源

- **MindSpore 官方**: <https://www.mindspore.cn/>
- **原始 PyTorch 版本**: [PowerFlowNet GitHub](https://github.com/...)
