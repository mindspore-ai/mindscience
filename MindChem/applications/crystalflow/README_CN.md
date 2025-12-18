# CrystalFlow

## 背景简介

理论晶体结构预测是通过计算的手段寻找物质在给定的外界条件下最稳定结构的重要手段。传统结构预测方法依赖在势能面上广泛的随机采样来寻找最稳定结构，然而，这种方法需要对大量随机生成的结构进行局域优化，而局域优化通常需要消耗巨大的第一性原理计算成本，尤其在模拟多元素复杂体系时，这种计算开销会显著增加，从而带来巨大的挑战。近年来，基于深度学习生成模型的晶体结构生成方法因其能够在势能面上更高效地采样合理结构而逐渐受到关注。这种方法通过从已有的稳定或局域稳定结构数据中学习，进而生成合理的晶体结构，与随机采样相比，不仅能够减少局域优化的计算成本，还能通过较少的采样找到体系的最稳定结构。采用神经常微分方程和连续变化建模概率密度的归一化流流模型，相比采用扩散模型方法的生成模型具有更加简洁、灵活、高效的优点。本方法基于流模型架构，发展了以CrystalFlow命名的晶体结构生成模型，在MP20等基准数据集上达到优秀的水平。

## 模型实现

### 硬件要求

- 支持 `CPU`、`Ascend` 两种后端，运行时可通过 `--device_target` 指定，默认 `CPU`（见 `train.py`）。

### 版本依赖

- 需要安装 `MindSpore >= 2.7.0`, `MindScience >= 0.8`。

### 安装

- 安装依赖包：`pip install -r requirement.txt`
- 安装 MindSpore：参考官方安装指南 `https://www.mindspore.cn/install`
- 安装 MindScience：参考 `https://atomgit.com/mindspore-lab/mindscience`

### 数据集

- 在[数据集链接](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/dataset/)下载相应的数据集文件夹与 `dataset_prop.txt` 属性文件，并放置于当前路径的 `dataset` 文件夹下（若不存在需手动创建）。

示例目录结构：

```txt
crystalflow
    └─dataset
            perov_5        钙钛矿数据集
            carbon_24      碳晶体数据集
            mp_20          晶胞内原子数最多为20的MP数据集
            mpts_52        晶胞内原子数最多为52的MP数据集
            dataset_prop.txt  数据集属性文件
```

### 核心代码实现

- 代码主要模块位于 `models` 与 `data` 文件夹：

```text
applications
  └── crystalflow
        ├── config.yaml                # 配置文件
        ├── train.py                   # 训练入口
        ├── evaluate.py                # 推理入口
        ├── compute_metric.py          # 评估入口
        ├── requirement.txt            # 环境依赖
        ├── train_pressure.py          # 压力条件训练入口
        ├── test_crystalflow.py        # 单元测试入口
        ├── data
        |     ├── data_utils.py        # 数据工具
        |     ├── dataset.py           # 构造数据集
        |     ├── dataloader.py        # 数据加载器封装
        |     └── crysloader.py        # 原始数据加载器
        ├── graph
        |     ├── graph.py             # 图结构构建
        |     └── loss.py              # 图相关损失
        └── models
              ├── conditioning.py      # 条件生成工具
              ├── cspnet.py            # 基于图神经网络的去噪器
              ├── cspnet_condition.py  # 条件生成网络层
              ├── diff_utils.py        # 模型工具
              ├── flow.py              # 流模型模块（核心）
              ├── flow_condition.py    # 条件生成的流模型
              ├── infer_utils.py       # 推理工具
              ├── lattice.py           # 晶格矩阵处理
              └── train_utils.py       # 训练工具
```

- 模型主体由 `CSPNet`（`models/cspnet.py`）与 `CSPFlow`（`models/flow.py`）构成：前者负责等变图神经网络特征提取与去噪，后者实现流式生成的潜变量变换与目标变量拟合；训练采用 `Adam` 优化器与 `L2LossMask`，并通过 `@ms.jit` 加速前向与训练步骤。

## 模型运行步骤

### 训练

- 修改 `config.yaml` 中的训练配置：
    - `dataset`：设置训练数据集与路径
    - `model`：设置网络层数、隐层维度、频率数等
    - `train.ckpt_dir` 与 `checkpoint.last_path`：设置权重保存目录与文件名
- 运行：

```bash
python train.py --device_target Ascend
```

### 推理

- 修改 `config.yaml` 的 `test` 字段，设置推理参数，特别是 `test.num_eval`（决定每个组分生成样本数，影响评估阶段）。
- 运行：

```bash
python evaluate.py
```

推理得到的晶体将保存在test.eval_save_path指定的文件中，文件中存储的内容为python字典，格式为：

```python
{
        'pred': [
                [晶体A sample 1, 晶体A sample 2, 晶体A sample 3, ... 晶体A sample num_eval],
                [晶体B sample 1, 晶体B sample 2, 晶体B sample 3, ... 晶体B sample num_eval]
                ...
        ]
        'gt': [
                晶体A ground truth,
                晶体B ground truth,
                ...
        ]
}
```

### 评估

- 将推理输出文件路径写入 `test.eval_save_path`。
- 确保 `num_evals` 与推理时的样本数设置一致或更小（例如推理为 20，则评估可设为 1–20）。
- 设置评估结果保存路径 `test.metric_dir`，运行：

```bash
python compute_metric.py
```

- 评估结果以 JSON 文件保存至 `metric_dir`，示例：

```json
{"match_rate": 0.6107671899181959, "rms_dist": 0.07492558322002925}
```

## 许可证

- 开源协议：`Apache License 2.0`
- 许可证链接：`http://www.apache.org/licenses/LICENSE-2.0`

## 引用

- 如果本项目对您的研究有帮助，请引用相关工作：
    - Luo X, Wang Z, Wang Q, et al. CrystalFlow: a flow-based generative model for crystalline materials[J]. Nature communications, 2025, 16: 9267.
