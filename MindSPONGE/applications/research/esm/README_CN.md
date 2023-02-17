[ENGLISH](README_EN.md)|简体中文

# ESM-IF1

ESM-IF1反向折叠模型用于通过骨架原子坐标预测蛋白质序列。我们在本项目中提供了如下脚本：1）对于给定结构进行序列采样设计；2）为给定结构的预测序列进行打分；3）模型训练脚本。ESM-IF1模型由几何不变输入处理层和sequence-to-sequence的transformer网络组成。该模型还使用跨度掩蔽训练去应对丢失的骨架坐标，因此可以预测部分掩蔽结构的序列。

![Illustration](illustration.png)

## 可用的模型和数据集

| 文件名            | 大小  | 描述                                | Model URL                                                    |
| ----------------- | ----- | ----------------------------------- | ------------------------------------------------------------ |
| `chain_set.jsonl` | 512MB | CATH v4.3数据集蛋白质骨架坐标和序列 | [下载链接](https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/chain_set.jsonl) |
| `splits.json`     | 197kB | CATH v4.3数据集划分                 | [下载链接](https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/splits.json) |

<details><summary>README目录</summary>
<!-- TOC -->

- [ESM-IF1](#ESM-IF1)
  - [环境配置](#环境配置)
    - [硬件环境与框架](#硬件环境与框架)
    - [Conda环境配置](#Conda环境配置)
  - [代码目录](#代码目录)
  - [运行示例](#运行示例)
    - [ESM-IF1模型训练](#ESM-IF1模型训练)
    - [ESM-IF1对于给定结构进行序列采样设计](#ESM-IF1对于给定结构进行序列采样设计)
    - [ESM-IF1序列评分](#ESM-IF1序列评分)
  - [致谢](#致谢)

<!-- /TOC -->

## 环境配置

### 硬件环境与框架

本项目基于[MindSPONGE](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE)生物计算库与[MindSpore](https://www.mindspore.cn/)AI框架开发，运行于Nvidia RTX3090和昇腾910，采用Mindspore深度学习框架。本项目可以通过自行配置运行环境使其部署于其他硬件环境。

本项目使用环境版本为：

mindspore-gpu 1.8.0 或 mindspore-ascend 1.8.1；

python 3.7；

### Conda环境配置

推荐使用一个新的conda环境。

使用如下命令安装新的conda环境并配置需要的库。

```text
conda create -n inverse python=3.7
conda activate inverse
conda install mindspore-gpu=1.8.1 cudatoolkit=11.1 -c mindspore -c conda-forge
conda install pip
pip install biotite
```

## 代码示例

<details><summary><font size=4 color="blue">代码目录</font></summary>

```bash
├── esm
    ├── illustration.png                // 模型结构图
    ├── README_EN.md                    // ESM-IF1相关英文说明
    ├── README_CN.md                    // ESM-IF1相关中文说明
    ├── src
        ├── args.json                   // 模型参数配置
        ├── data.py                     // 数据处理
        ├── features.py                 // 特征提取相关脚本
        ├── gvp_encoder.py              // gvp编码器脚本
        ├── gvp_modules.py              // gvp模块
        ├── gvp_transformer.py          // gvp_transformer模块
        ├── gvp_transformer_encoder.py  // gvp_transformer编码器模块
        ├── gvp_utils.py                // gvp所需功能函数
        ├── inspector.py                // inspector模块
        ├── message_passing.py          // 消息传递模块
        ├── modules.py                  // 模型所需模块
        ├── multihead_attention.py      // 多头注意力机制模块
        ├── pretrained.py               // 预训练脚本
        ├── transformer_decoder.py      // gvp_transformer解码器模块
        ├── transformer_layer.py        // transformer层
        ├── util.py                     // 模型所需功能函数
    ├── sample_sequences.py             // 对于给定结构进行序列采样设计
    ├── score_log_likelihoods.py        // 序列评分
    ├── train.py                        // 训练脚本
```

</details>

### ESM-IF1模型训练

可以用如下命令训练模型：

```bash
用法：python train.py --epochs 100

选项：
--epochs        模型训练代数
```

### ESM-IF1对于给定结构进行序列采样设计

对于 PDB或mmCIF格式的蛋白质结构，使用`sample_sequences.py` 脚本进行序列预测。输入文件可以是 `.pdb` 或`.cif`作为后缀。

例如，可以在`esm` 目录下，使用如下命令对高尔基酪蛋白激酶结构（PDB [5YH2](https://www.rcsb.org/structure/5yh2)；[PDB Molecule of the Month from January 2022](https://pdb101.rcsb.org/motm/265)）的3个序列设计进行取样：

```bash
用法：python sample_sequences.py data/5YH2.pdb
    --chain C --temperature 1 --num-samples 3
    --outpath output/sampled_sequences.fasta

选项：
--chain        蛋白质链种类
--temperature  采样温度
--num-samples  采样数量
--outpath      输出保存路径
```

采样序列将以fasta格式保存到指定的输出文件中。

温度参数控制序列采样的概率分布锐度。更高的采样温度产生更多样的序列，但可能具有更低的天然序列恢复率。默认采样温度为1。为了优化天然序列恢复，建议使用低温采样，如1e-6。

### ESM-IF1序列评分

使用`score_log_likelihoods.py`脚本对给定结构下的预测序列的条件对数似然进行评分。

例如，可以在`esm` 目录下，使用如下命令，根据`data/5YH2.pdb`的蛋白质结构对`data/5YH2_mutated_seqs.fasta`目录下的序列打分：

```bash
用法：python score_log_likelihoods.py data/5YH2.pdb \
    data/5YH2_mutated_seqs.fasta --chain C \
    --outpath output/5YH2_mutated_seqs_scores.csv \
    --pdbfile src/data/5YH2.pdb --seqfile src/data/5YH2_mutated_seqs.fasta

选项：
--chain        蛋白质链种类
--outpath      输出保存路径
--pdbfile      pdb数据文件路径
--seqfile      序列数据文件路径
```

条件对数似然以csv格式保存在指定的输出路径中。

输出值是序列中所有氨基酸的平均对数似然。

## 致谢

ESM-IF1使用或参考了以下开源工具：

- [AlphaFold2](https://github.com/deepmind/alphafold)
- [Biotite](https://www.biotite-python.org/install.html)

我们感谢这些开源工具所有的贡献者和维护者！

