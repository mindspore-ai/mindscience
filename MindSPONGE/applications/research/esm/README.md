# 使用ESM-IF1进行反向折叠

ESM-IF1反向折叠模型用于通过骨架原子坐标预测蛋白质序列。我们在本项目中提供了如下脚本：1）对于给定结构进行序列采样设计；2）为给定结构的预测序列进行打分；3）模型训练脚本。ESM-IF1模型由几何不变输入处理层和sequence-to-sequence的transformer网络组成。该模型还使用跨度掩蔽训练去应对丢失的骨架坐标，因此可以预测部分掩蔽结构的序列。

![Illustration](illustration.png)

## 环境

本项目运行于Nvidia RTX3090，采用Mindspore深度学习框架。本项目可以通过自行配置运行环境使其部署于其他硬件环境。

本项目使用环境版本为：

mindspore-gpu 1.8.0；

python 3.7；

## 组织架构

- src：数据处理和模型脚本；
- score_log_likelihoods：为给定结构的预测序列进行打分；
- sample_sequences：对于给定结构进行序列采样设计；
- train：esm的模型训练脚本.

## Conda 环境配置

推荐使用一个新的conda环境。

使用如下命令安装新的conda环境并配置需要的库。

```text
conda create -n inverse python=3.7
conda activate inverse
conda install mindspore-gpu=1.8.1 cudatoolkit=11.1 -c mindspore -c conda-forge
conda install pip
pip install biotite
```

## 快速开始

### 对于给定结构进行序列采样设计

对于 PDB或mmCIF格式的蛋白质结构，使用`sample_sequences.py` 脚本进行序列预测。输入文件可以是 `.pdb` 或`.cif`作为后缀。

例如，可以在`esm` 目录下，使用如下命令对高尔基酪蛋白激酶结构（PDB [5YH2](https://www.rcsb.org/structure/5yh2)；[PDB Molecule of the Month from January 2022](https://pdb101.rcsb.org/motm/265)）的3个序列设计进行取样：

```text
python sample_sequences.py data/5YH2.pdb \
    --chain C --temperature 1 --num-samples 3 \
    --outpath output/sampled_sequences.fasta
```

采样序列将以fasta格式保存到指定的输出文件中。

温度参数控制序列采样的概率分布锐度。更高的采样温度产生更多样的序列，但可能具有更低的天然序列恢复率。默认采样温度为1。为了优化天然序列恢复，建议使用低温采样，如1e-6。

### 序列评分

使用`score_log_likelihoods.py`脚本对给定结构下的预测序列的条件对数似然进行评分。

例如，可以在`esm` 目录下，使用如下命令，根据`data/5YH2.pdb`的蛋白质结构对`data/5YH2_mutated_seqs.fasta`目录下的序列打分：

```text
python score_log_likelihoods.py data/5YH2.pdb \
    data/5YH2_mutated_seqs.fasta --chain C \
    --outpath output/5YH2_mutated_seqs_scores.csv
```

条件对数似然以csv格式保存在指定的输出路径中。

输出值是序列中所有氨基酸的平均对数似然。

## 数据划分

本项目的训练数据为CATH v4.3数据集，可通过以下链接获得：

- [Backbone coordinates and sequences](https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/chain_set.jsonl)
- [Split](https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/splits.json)
