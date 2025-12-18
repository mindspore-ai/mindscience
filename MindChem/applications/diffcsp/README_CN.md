# DiffCSP

## 背景简介

DiffCSP 是一种基于扩散模型的深度生成框架，用于解决晶体结构预测这一基础科学难题。其核心思想是将寻找稳定晶体结构的过程转化为一个生成问题：模型通过学习大规模已知晶体数据中的分布规律，仅根据材料的化学成分（原子种类与比例），即可直接、快速地生成合理的三维原子结构（包括晶格和原子坐标）。

与传统依赖大量量子力学计算的结构预测方法相比，DiffCSP 的关键创新在于采用周期性 E(3)-等变图神经网络，并显式考虑平移、旋转与周期性对称性，确保生成结构严格遵守物理约束。借助这一设计，DiffCSP 能够高效探索庞大的晶体构型空间，以远低于第一性原理方法的计算成本获得高质量候选结构，为新材料的加速发现与设计提供有力工具。

## 模型实现

### 硬件要求

- 支持 `Ascend` 后端，运行时可通过 `--device_target` 指定，默认 `Ascend`（见 `train.py`）。

### 版本依赖

- 需要安装 `MindSpore >= 2.7.0`。
- 需要安装 `MindScience`，以提供等变计算相关的基础组件。

### 安装

- 安装 MindSpore：参考官方安装指南 `https://www.mindspore.cn/install`
- 安装 MindScience：参考 `https://atomgit.com/mindspore-lab/mindscience`
- 安装依赖包：`pip install -r requirement.txt`

### 数据集

- 在[数据集链接](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/dataset/)下载相应的数据集文件夹与 `dataset_prop.txt` 属性文件，并放置于当前路径的 `dataset` 文件夹下（若不存在需手动创建）。

示例目录结构：

```txt
diffcsp
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
  └── diffcsp
        ├── README.md                   # 中文说明
        ├── README_EN.md                # 英文说明
        ├── config.yaml                 # 配置文件
        ├── train.py                    # 训练入口
        ├── evaluate.py                 # 推理入口
        ├── compute_metric.py           # 评估入口
        ├── requirement.txt             # 环境依赖
        ├── data
        |     ├── data_utils.py         # 数据集处理工具
        |     ├── dataset.py            # 读取并构造数据集
        |     ├── dataloader.py         # 数据加载器封装
        |     └── crysloader.py         # 原始数据加载器
        └── models
              ├── cspnet.py             # 基于图神经网络的去噪器
              ├── diffusion.py          # 扩散模型模块
              ├── diff_utils.py         # 模型工具
              ├── infer_utils.py        # 推理工具
              ├── train_utils.py        # 训练工具
              ├── graph.py              # 图结构与邻接构建工具
              └── loss.py               # 损失模块
```

- 模型主体由 `CSPNet`（`models/cspnet.py`）与 `CSPDiffusion`（`models/diffusion.py`）共同构成：前者为周期性 E(3)-等变去噪网络，负责对晶格与原子坐标进行表示与去噪；后者实现扩散过程中的前向/反向采样与晶体结构生成。训练阶段采用 `Adam` 优化器与 `L2LossMask` 损失（`models/loss.py`），并通过 `@ms.jit` 加速前向与训练步骤。

## 模型运行步骤

### 训练

- 确保已完成以下准备：
    - 安装 MindSpore 及依赖包；
    - 下载并整理 `dataset` 数据集目录（见上文“数据集”部分）；
    - 根据任务需求修改 `config.yaml` 中的训练参数：
        - `dataset`：设置训练数据集名称与路径；
        - `train.epoch_size`：训练轮数；
        - `model`：设置去噪器网络层数、隐层维度、频率数等；
        - `train.ckpt_dir` 与 `checkpoint.last_path`：设置权重保存目录与文件名；
        - 其它训练参数见 `train`、`checkpoint` 等字段。
- 在 `diffcsp` 目录下执行：

```bash
python train.py
```

### 推理

- 将需要加载的权重路径写入 `config.yaml` 中 `checkpoint.last_path` 字段。预训练模型可从[预训练模型链接](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/pre-train)获取。
- 修改 `config.yaml` 的 `test` 字段，设置推理参数，特别是 `test.num_eval`（决定每个组分生成样本数，对后续评估阶段至关重要）。
- 在 `diffcsp` 目录下执行：

```bash
python evaluate.py
```

推理得到的晶体将保存在 `test.eval_save_path` 指定的文件中，文件中存储的内容为 Python 字典，格式为：

```python
{
        'pred': [
                [晶体A sample 1, 晶体A sample 2, 晶体A sample 3, ... 晶体A sample num_eval],
                [晶体B sample 1, 晶体B sample 2, 晶体B sample 3, ... 晶体B sample num_eval]
                ...
        ],
        'gt': [
                晶体A ground truth,
                晶体B ground truth,
                ...
        ]
}
```

### 评估

- 将推理输出文件路径写入 `config.yaml` 中的 `test.eval_save_path`。
- 确保 `num_evals` 与推理时生成样本数设置一致或更小（例如推理时 `num_evals=20`，则评估时 `num_evals` 可设为 1–20；若推理时 `num_evals=1`，则评估时也只能设为 1）。
- 设置评估结果保存路径 `test.metric_dir`，在 `diffcsp` 目录下执行：

```bash
python compute_metric.py
```

评估结果将以 JSON 文件形式保存在 `metric_dir` 指定目录下，示例：

```json
{"match_rate": 0.985997357992074, "rms_dist": 0.013073775170360118}
```

## 许可证

- 开源协议：`Apache License 2.0`
- 许可证链接：`http://www.apache.org/licenses/LICENSE-2.0`

## 引用

- 如果本项目对您的研究有帮助，请引用相关工作，例如：
    - Jiao Rui and Huang Wenbing and Lin Peijia, et al. Crystal structure prediction by joint equivariant diffusion[J]. Advances in Neural Information Processing Systems, 2024, 36.
