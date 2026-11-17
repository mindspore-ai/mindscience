# Bucket Dataset Generator

## 背景介绍

在图神经网络（GNN）的训练场景中，不同样本的图规模（节点数量）往往差异显著。若采用固定批次大小进行训练，小图样本会被过度填充以对齐大图，导致大量计算资源浪费与内存冗余。为解决这一问题，本项目提供了一套基于桶（Bucket）策略的动态分桶数据集生成工具，核心思路如下：

- **分桶机制**：按照节点数量将样本划分为多个桶区间，每个桶配置独立的批次大小，使大图样本以更小批次处理、小图样本以更大批次处理，从而在填充与计算效率之间取得平衡。
- **动态迭代**：训练阶段通过随机选择列表控制每个迭代步从哪个桶中取数据，确保各桶数据均匀消耗，并在迭代结束后自动处理残余数据。
- **MindSpore 集成**：基于 MindSpore `GeneratorDataset` 接口构建，支持分布式多卡并行数据加载与分片，无缝对接 MindSpore 训练流程。

本项目同时支持 **训练（train）**、**测试（test）** 和 **推理（infer）** 三种任务模式，并提供 HDF5 与 NPY 两种数据格式的加载能力。
根据实际需求，当前仅针对训练模式提供分桶能力，而测试/推理则采取常规固定batch_size的形式以供对比。

## 项目结构

```
GNN_BucketDatasetOnlinePadding/
├── configs/
│   └── config.yaml            # 训练/测试/推理数据集与模型参数配置文件
├── src/
│   ├── __init__.py            # 包初始化，导出核心类与函数
│   ├── BucketDatasetGenerator.py  # 分桶数据集迭代器，实现动态分桶批次生成
│   ├── padding.py             # 批内对齐：声明式列登记表 + pad/拼接/边归属
│   ├── DataGenerator.py       # 数据生成器，负责单样本读取与特征重组
│   └── tools.py               # 工具函数：YAML 加载、NPY 目录加载
├── tests/
│   ├── test_padding.py               # padding 模块单测（纯 numpy）
│   ├── test_package_data.py          # 批组包（_package_data）与列校验单测
│   ├── test_bucket_state_machine.py  # 桶迭代状态机单测（stub 数据集，纯 numpy）
│   └── test_tools.py                 # load_npy/load_yaml 单测（依赖 pyyaml）
├── main.py                    # 主入口：构建 MindSpore 分桶数据集主流程，可供后续训练集成
├── requirements.txt           # Python 依赖库及版本声明
├── README_CN.md               # 项目中文说明文档
└── README.md                  # 项目英文说明文档
```

各模块职责说明：

| 模块 | 说明 |
|------|------|
| `main.py` | 程序主入口，解析配置文件，根据任务类型（train/test/infer）调用对应数据构建流程 |
| `src/DataGenerator.py` | `DataGenerator` 类，从 HDF5 或 NPY 数据源逐样本读取节点/边特征，并作相应处理 |
| `src/BucketDatasetGenerator.py` | `BucketDatasetGenerator` 类，按节点数分桶、动态填充与拼接批次，管理残余数据回收 |
| `src/padding.py` | 批内对齐的唯一实现：`NODE_FILLS`/`EDGE_ATTRS` 声明哪些列属于节点轴/边轴，`pad_nodes`/`edge_batch_ids` 完成对齐 |
| `src/tools.py` | 提供 `load_yaml`（YAML 配置解析）、`load_npy`（NPY 目录批量加载）两个工具函数 |
| `configs/config.yaml` | YAML 格式的统一配置文件，涵盖模型参数、训练/测试/推理数据集参数 |
| `tests/` | 覆盖 padding、批组包、桶迭代状态机与 `tools` 的单元测试，见[单元测试](#单元测试) |

## 环境要求

### 硬件环境

| 硬件 | 说明 |
|------|------|
| Huawei Ascend 处理器 | 推荐 Huawei Ascend 910 系列，用于分布式训练 |
| CPU | 可用于单卡测试与推理 |

### 软件环境

| 软件 | 版本要求 |
|------|----------|
| Python | >= 3.9 |
| MindSpore | >= 2.7.0 |
| NumPy | >= 1.26.0 |
| h5py | >= 3.7.0 |
| PyYAML | >= 6.0 |



## 安装与配置

### 1. 安装 MindSpore

请根据目标硬件平台选择对应的 MindSpore 安装方式，详见 [MindSpore 官方安装指南](https://www.mindspore.cn/install)。

Huawei Ascend 环境示例：

```bash
pip install mindspore==2.7.0
```

### 2. 安装其余依赖

```bash
pip install -r requirements.txt
```

或逐个安装：

```bash
pip install numpy>=1.26.0 h5py>=3.7.0 pyyaml>=6.0
```

### 3. 准备数据文件

将训练/测试/推理数据文件放置于指定路径。支持以下格式：

- **HDF5 文件**（`.h5`）：单个文件包含所有特征键（`x0`、`node_feature1` ~ `node_feature3`、`edge_feature1`、`edge_feature2`、`x_vd`）
- **NPY 目录**：目录下每个 `.npy` 文件对应一个特征键，文件名即为键名

特征键名支持自定义。

在 `configs/config.yaml` 中修改各数据集的 `path` 字段指向实际数据路径。

### 4. 配置参数

编辑 `configs/config.yaml`，按需调整以下参数：

```yaml
model:
    x_features: 7          # 输入特征维度数
    output: 3               # 输出特征维度数

data:
    use_mock_data: True     # 使用内存随机数据构建数据集而非读取 `path`；接入真实数据时置为 False
    train:
        role: train
        path: "/home/data/train.h5"
        batch_size: 8
        bucket_boundaries: [10, 20, 30]   # 桶上界列表
        bucket_batch_size: [16, 8, 4]     # 各桶对应批次大小
        sample_num_by_node: {5: 20, 11: 50, 15: 50, 25: 80}  # 节点数-样本数映射
        pad_quantum: 1                    # pad 目标向上取整到该值的整数倍，1 表示恰好 pad 到批内最大值
```

## 使用方法

### 训练模式（分桶批次）

```bash
python main.py
```

程序将读取 `configs/config.yaml`，构建分桶训练数据集。训练模式下 `BucketDatasetGenerator` 会自动：

1. 按节点数量将每个样本分配至对应桶
2. 依据随机选择列表从满足批次条件的桶中取数据
3. 把节点级列（`x0`、`node_fea1~3`、`labels`）填充至批内最大节点数，样本 `i` 固定占用第 `i * max_node` 行起的槽位
4. 把边级列（`edge_fea1`、`edge_fea2`、`mean_edge`）按原顺序拼接，并用 `batch_id` 记录每条边属于哪个样本
5. 迭代结束后回收残余数据

节点级/边级由 `src/padding.py` 里的 `NODE_FILLS`、`EDGE_ATTRS` 两张表声明，在表里登记一列即可完成它的
组批，无需其他配套修改。样本保留自身节点数直到组批时刻，批内参差由 `_package_data` 吸收，因此输入数据
不必预先 pad 到桶边界；`pad_quantum` 大于 1 时以少量显存换取更少的构图次数。

日志示例：
```
Start Load Dataset
Task: train
Dataset Bucket Boundary: [10, 20, 30], Batch Num(estimated): 41
Create bucket train dataset successfully
```


### 分布式训练

如需多卡并行训练，需先启动 MindSpore 分布式运行环境，程序会自动调用 `get_rank()` 与 `get_group_size()` 进行数据分片。

### 测试 / 推理模式

修改 `configs/config.yaml` 中对应数据集的 `role` 字段为 `test` 或 `infer`，然后：

```bash
python main.py
```

测试与推理模式采用固定批次大小，分别调用 `test_collate` 与 `infer_collate` 进行数据拼接与 `edge_batch_id` 构建。

## 单元测试

在本应用目录下运行全部测试：

```bash
python -m unittest discover -s tests
```

| 测试文件 | 覆盖范围 | 依赖 |
|----------|----------|------|
| `tests/test_padding.py` | `pad_to_max` / `pad_nodes` / `edge_batch_ids` / `round_up` | numpy |
| `tests/test_package_data.py` | `_package_data` 批布局与列校验 | numpy |
| `tests/test_bucket_state_machine.py` | 桶迭代：样本恰消费一次、各桶批次大小、越界校验、实际批次数超过 1.15 倍估计值的回归 | numpy（基于 stub 数据集，无需 MindSpore） |
| `tests/test_tools.py` | `load_npy` 键名解析、`load_yaml` | numpy + pyyaml |

## 许可证说明

本项目遵循 **Apache License 2.0** 开源协议。

核心条款：

- **授权**：允许自由使用、修改、分发与商业化，无需额外授权
- **归属**：分发时必须保留原始版权声明与许可证文本
- **专利**：许可证涵盖贡献者授权的专利使用权
- **免责**：软件按"原样"提供，不承担任何明示或暗示的担保责任
- **责任限制**：作者/贡献者对因使用本软件导致的任何损失不承担责任

完整许可证文本请参阅项目源码头部引用的 [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)。
