# Molecular-graph-BERT

MG-BERT利用无监督原子表示学习进行分子性质预测。

我们在本项目中提供了如下脚本：1）掩蔽原子预测预训练；2）指定任务分类；3）指定任务回归。

![network](network.png)

## 环境

本项目运行于Nvidia RTX3090，采用Mindspore深度学习框架。本项目可以通过自行配置运行环境使其部署于其他硬件环境。

本项目使用环境版本为：

mindspore-gpu 1.8.0；

python 3.7；

## 组织架构

- src：数据处理和模型脚本；
- pretrain：为掩蔽原子预测进行预训练；
- classification ：进行分类预测；
- regression：进行回归预测.

## Conda 环境配置

推荐使用一个新的conda环境。

使用如下命令安装新的conda环境并配置需要的库。

```text
conda create -n mgbert python=3.7
conda activate mgbert
conda install mindspore-gpu=1.8.1 cudatoolkit=11.1 -c mindspore -c conda-forge
conda install -c openbabel openbabel
```

## 快速开始

### 对于掩蔽原子预测进行预训练

预训练阶段使用大量的未标记分子来挖掘分子中的未标记信息，可用如下命令进行预训练：

```text
python pretrain.py --path='data/chembl_31_chemreps.txt' --trained_epoch=100 --vocab_size=17
```

### 分类

使用`classfication.py`脚本对不同任务进行分子性质预测分类。

例如，可用如下命令对`Pgp_sub`任务进行分类：

```text
python classfication.py --task='Pgp_sub' --pretraining=0 --trained_epoch=100 --vocab_size=17
```

### 回归

使用`regression.py`脚本对不同任务进行分子性质回归预测。

例如，可用如下命令对`logS`任务进行回归预测：

```text
python regression.py --task='logS' --pretraining=0 --trained_epoch=100 --vocab_size=17
```

## 数据

预训练阶段所用数据从CHeMBL数据库获得，通过随机抽取数据库中170万个化合物作为训练数据。微调阶段使用从ADMETlab和MoleculeNet中收集的16个数据集（8个回归，8个分类）对MG-BERT进行训练和评估。将数据集按8：1：1的比列分为训练、验证和测试数据集。并使用SMILES长度分层抽象，使数据集的分裂更加均匀。

预训练所用数据可通过以下链接中chembl_31_chemreps.txt.gz获得：

- https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_31/

回归和分类所用数据可通过以下链接获得：

- https://gitee.com/lytgogogo/project_data/tree/master/data

