# Protenix (MindSpore)

[English Version](./README.md)

![result](./img/protenix_res_sample.png)

**Protenix** 是由字节跳动开发的[AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w) 的开源复现版本，基于PyTorch实现，[代码仓地址](https://github.com/bytedance/Protenix)。本项目为**MindSpore** 复现版本。

它专为高精度结构预测而设计，旨在为计算生物学社区提供可扩展的研究工具。Protenix 为研究人员提供了一个强大的平台，用于预测包括蛋白质、核酸、小分子、离子和修饰残基在内的生物分子结构。

## 📖 模型概述

Protenix 是当前开源生物分子结构预测领域的最强模型，能够以前所未有的精度预测复杂生物分子组装体的三维结构。Protenix 在 MindSpore 框架上忠实地复现了这一能力，支持：

- **多组分结构预测**：预测包含蛋白质、RNA、DNA、配体、离子和修饰残基的结构
- **高精度**：在基准数据集上实现与 AlphaFold 3 相当的精度
- **可扩展训练**：利用 MindSpore 的分布式训练能力实现高效的模型训练

**主要特性：**

- 完整复现 Protenix 架构，包括 Pairformer、Diffusion、Confidence等模块
- 支持多序列比对（MSA）
- 可配置的扩散采样，支持自定义循环次数、步数和样本数
- 支持在自定义数据集上进行训练和微调

## 🌟 相关项目

- **[MindScience](https://atomgit.com/mindspore-lab/mindscience)**：基于 MindSpore 的科学计算套件
- **[AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w)**：AlphaFold 3 原始论文

## 🛠 安装

### 硬件支持

- Atlas 800T A2

### 软件依赖

- Python >= 3.11
- MindSpore >= 2.7.0
- CANN >= 8.2.RC1
- 其他依赖项见 `requirements.txt`

### 步骤 1：克隆仓库

```bash
git clone https://gitcode.com/mindspore-lab/mindscience.git
cd mindscience/MindSPONGE/applications/protenix
```

### 步骤 2：安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 3：下载模型权重

从以下链接下载[模型权重](https://tools.mindspore.cn/dataset/workspace/mindspore_ckpt/ckpt/Protenix/ms_model_v0.5.0.ckpt)并将其放在`./release_data/checkpoint`下。

### 步骤 4：设置环境变量

运行一下命令设置PYTHONPATH：

```bash
source set_path.sh
```

## 🚀 推理

### 基本用法

使用默认参数运行推理：

```bash
python inference.py \
  --seeds 42 \
  --dump_dir ./output \
  --input_json_path /PATH/TO/INPUT/FILE/input.json \
  --use_msa true
```

### 输入格式

创建一个指定生物分子系统的输入 JSON 文件。示例：

```json
[
  {
    "sequences": [
      {
        "proteinChain": {
          "sequence": "SEFEKLRQTGDELVQAFQRLREIFDKGDDDSLEQVLEEIEELIQKHRQLFDNRQEAADTEAAKQGDQWVQLFQRFREAIDKGDKDSLEQLLEELEQALQKIRELAEKKN",
          "count": 1
        }
      }
    ],
    "name": "5tgy"
  }
]
```

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--seeds` | 随机种子（可重复性） | 42 |
| `--dump_dir` | 预测结果输出目录 | 无(该参数必须) |
| `--input_json_path` | 输入 JSON 文件路径 | 无(该参数必须) |
| `--use_msa` | 是否使用 MSA 搜索 | `true` |
| `--n_samples` | 生成样本数量 | 5 |
| `--load_checkpoint_path` | 模型权重路径 | `"./release_data/checkpoint/ms_model_v0.5.0.ckpt"` |

### 高级选项

**控制采样：**

```bash
python inference.py \
  --seeds 42 \
  --dump_dir ./output \
  --input_json_path /PATH/TO/INPUT/FILE/input.json \
  --n_sample 10
```

**不使用 MSA 运行（可能造成精度降低）：**

```bash
python inference.py \
  --seeds 42 \
  --dump_dir ./output \
  --input_json_path /PATH/TO/INPUT/FILE/input.json \
  --use_msa false
```

### 输出文件

推理会在指定的 `dump_dir` 目录中生成以下输出：

1. **cif 文件**：最终预测的结构（每个样本一个）
    - 格式：`sample_<N>.pdb`
    - 包含所有原子的三维坐标

2. **置信度分数**：
    - `confidence_scores.json`：每个残基的 pLDDT 分数和 PAE 矩阵
    - pLDDT 越高表示置信度越高（范围：0-100）

## 🧬 训练

### 数据预处理

对于下载到的cif格式文件数据需要进行预处理后用于模型训练，预处理脚本调用方式如下：

```bash
python scripts/prepare_training_data.py -i /PATH/TO/MMCIF/FILES -o /PATH/TO/OUTPUT/DATA.csv.gz -b /PATH/TO/PROCESSED/DATA -d
```

如搜索MSA:

```bash
python scripts/prepare_training_data.py -i /PATH/TO/MMCIF/FILES -o /PATH/TO/OUTPUT/DATA.csv.gz -b /PATH/TO/PROCESSED/DATA -d --use_msa
```

### 数据预处理参数说明

| 参数 | 说明 | 默认值 |
|------|------|-----------|
| `-i` | 输入数据目录 | 无(该参数必选) |
| `-o` | csv文件输出路径 | 无(该参数必选) |
| `-b` | 数据输出目录 | 无(该参数必选) |
| `-d` | 是否进行额外出处理（包括删除水及氢原子等） | 无需赋值 |
| `--use_msa` | 是否进行MSA搜索 | 无需赋值 |
| `-m` | MSA搜索结果路径 | None |

### 模型训练

训练逻辑主要包含在 `train.py` 中。

```bash
python train.py --run_name protenix_train --seed 42 --base_dir ./output --diffusion_batch_size 48 --checkpoint_interval 400 --train_crop_size 384 --lr 0.001 --data.msa.enable false --load_checkpoint_path /PATH/TO/YOUR/CHECKPOINT --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 --data.test_sets weightedPDB_before2109_wopb_nometalc_0925 --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list /PATH/TO/YOUR/DATASET/PDB_LIST.txt --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.mmcif_dir /PATH/TO/YOUR/MMCIF_DIR --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.indices_fpath /PATH/TO/YOUR/INDICES_FILE.csv --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.bioassembly_dict_dir /PATH/TO/BIOASSEMBLY_DIR
```

### 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|-----------|
| `--run_name` | 训练运行名称 | 无(该参数必选) |
| `--seed` | 随机种子（可重复性） | `42` |
| `--base_dir` | 输出基础目录 | 无(该参数必选) |
| `--diffusion_batch_size` | 扩散训练批次大小 | `48` |
| `--checkpoint_interval` | 检查点保存间隔步数 | `-1` |
| `--train_crop_size` | 训练样本裁剪大小 | `384` |
| `--lr` | 学习率 | `0.0018` |
| `--load_checkpoint_path` | 检查点路径（用于微调） | `""` |
| `--data.train_sets` | 训练集名称 | `weightedPDB_before2109_wopb_nometalc_0925`, 具体可参考configs.configs_data |
| `--data.test_sets` | 测试集名称 | `recentPDB_1536_sample384_0925`,  具体可参考configs.configs_data |
| `--data.msa.enable` | 启用/禁用 MSA 特征 | `True` |
| `--data.msa.prot.pdb_mmseqs_dir` | MSA文件路径 | `DATA_ROOT_DIR/mmcif_msa` |
| `--data.msa.prot.seq_to_pdb_idx_path` | seq_to_pdb_idx路径 | `DATA_ROOT_DIR/seq_to_pdb_index.json` |

### 数据集路径配置

训练命令需要指定数据集文件的路径：

- `--data.[DATASET_NAME].base_info.pdb_list`：PDB 列表文件路径
- `--data.[DATASET_NAME].base_info.mmcif_dir`：mmCIF 文件目录
- `--data.[DATASET_NAME].base_info.indices_fpath`：索引 CSV 文件路径
- `--data.[DATASET_NAME].base_info.bioassembly_dict_dir`：生物组装数据目录

将 `[DATASET_NAME]` 替换为数据集名称（例如：`weightedPDB_before2109_wopb_nometalc_0925`）。

## 许可证

本项目采用 Apache 2.0 许可证。

## 引用 Protenix

如果您在研究中使用了 Protenix，请引用：

```bibtex
@article{chen2025protenix,
  title={Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
  author={Chen, Xinshi and Zhang, Yuxuan and Lu, Chan and Ma, Wenzhi and Guan, Jiaqi and Gong, Chengyue and Yang, Jincai and Zhang, Hanyu and Zhang, Ke and Wu, Shenghao and Zhou, Kuangqi and Yang, Yanping and Liu, Zhenyu and Wang, Lan and Shi, Bo and Shi, Shaochen and Xiao, Wenzhi},
  year={2025},
  doi = {10.1101/2025.01.08.631967},
  journal = {bioRxiv}
}
@article{abramson2024accurate,
  title={Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  author={Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J and Bambrick, Joshua and others},
  journal={Nature},
  volume={630},
  number={8016},
  pages={493--500},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
@article{ahdritz2024openfold,
  title={OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization},
  author={Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O’Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccol{\`o} and others},
  journal={Nature Methods},
  volume={21},
  number={8},
  pages={1514--1524},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  volume={19},
  number={6},
  pages={679--682},
  year={2022},
  publisher={Nature Publishing Group US New York}
}
```
