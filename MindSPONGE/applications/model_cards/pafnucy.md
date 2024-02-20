# Pafnucy

## 模型介绍

[Pafnucy](https://academic.oup.com/bioinformatics/article/34/21/3666/4994792)是一种用于预测蛋白质-配体复合物亲和性的深度卷积神经网络。使用三维网格表征复合物，在模型中使用3D卷积生成该表征的特征图，以相同的方式处理蛋白质和配体的原子。

Pafnucy模型由卷积模块和线性模块两部分组成，层与层之间的连接类型不同。卷积模块由一个3D卷积层和最大池化层构成。Pafnucy使用了三个分别带有64，128，256个过滤器的卷积层的卷积模块，将最后一个模块的输出结果平坦化之后作为输入进入全连接层的模块中。

## 使用限制

该模型依赖于软件Open Babel，在使用前需提前安装openbabel-3.1.1，并且使用pip install的方式安装Open Babel对应版本python包。

Open Babel依赖于低版本python，所以安装前请确保 `python <= 3.7.16`。

可使用conda安装Open Babel软件，具体安装指令如下：

```bash
conda install conda-forge::openbabel
```

可在终端使用如下指令验证Open Babel是否安装成功：

```bash
obabel --help
```

## 数据集

模型所使用数据集为PDBBind v2016，数据集大小约为2.5G。

- [Index files of PDBbind](http://www.pdbbind.org.cn/download/PDBbind_2016_plain_text_index.tar.gz)
- [Protein-ligand complexes: The general set minus refined set](http://www.pdbbind.org.cn/download/pdbbind_v2016_general-set-except-refined.tar.gz)
- [Protein-ligand complexes: The refined set](http://www.pdbbind.org.cn/download/pdbbind_v2016_refined.tar.gz)
- [Protein-protein complexes](http://www.pdbbind.org.cn/download/PDBbind_v2016_PP.tar.gz)
- [Ligand molecules in the general set (Mol2 format)](http://www.pdbbind.org.cn/download/PDBbind_v2016_mol2.tar.gz)
- [Ligand molecules in the general set (SDF format)](http://www.pdbbind.org.cn/download/PDBbind_v2016_sdf.tar.gz)
- [pdbbind_v2013_core_set.tar.gz](http://www.pdbbind.org.cn/download/pdbbind_v2013_core_set.tar.gz)

### 数据集下载

用户可以登录[PDBbind-CN Database](http://www.pdbbind.org.cn/download.php)根据自己的需求进行数据下载，也可以根据[数据集](https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/model_cards/pafnucy.md#%E6%95%B0%E6%8D%AE%E9%9B%86)提供的链接下载数据到同一个文件夹中，并使用如下命令进行解压。

```bash
mkdir PDBbind_2016_plain_text_index
tar -zxvf PDBbind_2016_plain_text_index.tar.gz -C PDBbind_2016_plain_text_index
tar -zxvf pdbbind_v2013_core_set.tar.gz
tar -zxvf pdbbind_v2016_general-set-except-refined.tar.gz
tar -zxvf PDBbind_v2016_mol2.tar.gz
tar -zxvf PDBbind_v2016_PP.tar.gz
tar -zxvf pdbbind_v2016_refined.tar.gz
tar -zxvf PDBbind_v2016_sdf.tar.gz
cp index/INDEX_core_data.2016 PDBbind_2016_plain_text_index/index
```

## 如何使用

```bash
import os
from mindsponge import PipeLine
from openbabel import pybel

# 小分子为mol2文件，蛋白质为pdb文件或mol2文件
pocket_path = {YOUR_POCKET_PATH}
ligand_path = {YOUR_LIGAND_PATH}
raw_data = [pocket_path, ligand_path]
pipe = PipeLine(name="Pafnucy")
pipe.set_device_id(0)
pipe.initialize("pafnucy_predict")
pipe.model.from_pretrained()
result = pipe.predict(raw_data)
print(result)
```

## 训练过程

训练只需向模型提供数据集所在路径，若该路径下不存在数据集，则模型会自动下载训练所需PDBBind数据集，之后进行训练。训练时需将config文件中的is_training修改为True。

```bash
from mindsponge import PipeLine
pipe = PipeLine(name="Pafnucy")
pipe.set_device_id(0)
pipe.initialize("pafnucy_training")
pipe.train({YOUR_DATA_PATH}, num_epochs=1)
```

## 引用

```bash
@article{10.1093/bioinformatics/bty374,
    author = {Stepniewska-Dziubinska, Marta M and Zielenkiewicz, Piotr and Siedlecki, Pawel},
    title = "{Development and evaluation of a deep learning model for protein–ligand binding affinity prediction}",
    journal = {Bioinformatics},
    volume = {34},
    number = {21},
    pages = {3666-3674},
    year = {2018},
    month = {05},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/bty374},
    url = {https://doi.org/10.1093/bioinformatics/bty374},
}
```
