# UFold

## 模型介绍

从核苷酸序列预测RNA的二级结构是基因组学中一个长期存在的问题。
传统的RNA二级结构预测算法主要基于热力学模型，通过自由能最小化，这强加了很强的先验假设，而且运行速度很慢。
UFold作为一种基于深度学习的方法，直接根据注释数据和碱基配对规则进行训练，
基于一种新的RNA序列的类图像表示法，通过完全卷积网络(FCNs)进行RNA二级结构预测。

## RNA的二级结构

![RNA的二级结构和矩阵表示](https://foruda.gitee.com/images/1690286617878764738/a7844666_13136640.png "RNA的二级结构和矩阵表示")

RNA的二级结构是指核酸聚合物内的**碱基对的相互作用**，可以被表示成反映核酸分子中配对的碱基的**接触矩阵**（contact matrix）。
其中A为`L*L`大小的二值对称矩阵，取值范围为{0,1}。
当`A_ij=1`时，代表碱基`x_i`与`x_j`之间存在一个碱基对（base pairing），
反之当`A_ij=0`则代表碱基`x_i`与`x_j`之间不存在碱基对的相互作用。

## 模型的输入输出

UFold所解决的任务为：给定一个输入序列，预测他们的碱基配对模态(base pairing patterns)，即RNA的二级结构：

- 输入：长度为L的核苷酸序列而转换成的，具有17个通道的`L*L`的矩阵。

- 输出：RNA的二级结构，用`L*L`的接触矩阵A表示。其中`A_ij`代表i位置和j位置存在碱基对的概率。

![UFold输入输出结构图](https://foruda.gitee.com/images/1690200483011251997/6d1e0790_13136640.png "UFold输入输出结构图")
假设，某个核苷酸的序列是`x = {x_1,x_2, x_3,……,x_L }`，其中`x_i∈{A, U, C, G}`。
则其可以被转化成Lx4的矩阵表示，其中每一行可以视为一个One-Hot Encoding，每一列有且仅有一个1（其他位置为0），表示对应的核苷酸种类。
然后，将该矩阵的每一列的两两间做外积（可以理解成Lx1的向量和1xL的向量的乘法），一共得到16个LxL的矩阵，可以视为一个`L*L`的矩阵的16个通道。
还会根据一个已知的配对规则输入碱基对之间的配对概率，作为第17个通道。
UFold将会将这个具有17个通道的`L*L`的矩阵作为模型的输入，并输出能反映RNA中是否存在碱基对的相互作用的`L*L`的矩阵，进而实现对RNA二级结构的预测。

## 数据集

UFold使用了多个基准数据集：

- RNAStralign，包含来自8个RNA家族的30 451个独特序列；

- ArchiveII，包含来自10个RNA家族的3975个序列，是最广泛使用的RNA结构预测性能基准数据集；

- bpRNA-1m，包含来自2588个家族的102 318个序列，是可用的最全面的RNA结构数据集之一；

- bpRNA new，源自Rfam 14.2，包含来自1500个新RNA家族的序列。

原始数据集ArchiveII，bpnew，TS0，TS1，TS2，TS3为bpseq格式数据文件，在使用前需要将原始bpseq格式数据文件处理成pickle文件，处理后的数据文件可从[网盘](https://pan.baidu.com/s/1y2EWQlZJhJfqi_UyUnEicw?pwd=o5k2)中下载，下载后将数据置于data文件夹下。

## 如何使用

推理可支持输入单个RNA的ct文件，也可以以文件夹作为路径输入，文件夹中存储所有ct文件。当输入为单个ct文件时，推理结果为单个预测结果，当输入为文件夹时，推理结果为文件夹下所有ct文件的推理结果，顺序与ct文件首字母排序顺序相同。

```bash
import collections
from mindsponge import PipeLine
from mindsponge.common.config_load import load_config
from mindsponge.pipeline.models.ufold.ufold_data import RNASSDataGenerator

RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
pipe = PipeLine(name = "UFold")
data_src = {YOUR_DATA_PATH}

# 第一次使用时未获取config文件，执行如下指令模型可自动下载config文件，后续使用可手动修改所需内容
# from mindsponge.pipeline.pipeline import download_config
# conf = download_config(pipe.config["ufold_config"], pipe.config_path + "ufold_config.yaml")

config_path = {YOUR_CONFIG_PATH}
conf = load_config(config_path)
conf.is_training = False
# 可选test_ckpt为'ArchiveII', 'bpnew', 'TS0', 'TS1', 'TS2', 'TS3', 'All'
conf.test_ckpt = 'All'
pipe.set_device_id(0)
pipe.initialize(conf=conf)
pipe.model.from_pretrained()
data = {/YOUR_DATA_PATH/xxx.ct}
# data = {/YOUR_DATA_PATH/}
result = pipe.predict(data)
print(result)
```

## 训练过程

```bash
import collections
from mindsponge import PipeLine
from mindsponge.common.config_load import load_config
from mindsponge.pipeline.models.ufold.ufold_data import RNASSDataGenerator

RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
pipe = PipeLine(name = "UFold")
config_path = {YOUR_CONFIG_PATH}
conf = load_config(config_path)
conf.is_training = True
# 训练集可为['ArchiveII', 'bpnew', 'TS0', 'TS1', 'TS2', 'TS3']中的一个或多个
conf.train_files = ['TS0']
pipe.set_device_id(1)
pipe.initialize(conf=conf)
pipe.train({YOUR_DATA_PATH}, num_epochs = 10)
```

## 引用

```bash
@article{10.1093/nar/gkab1074,
    author = {Fu, Laiyi and Cao, Yingxin and Wu, Jie and Peng, Qinke and Nie, Qing and Xie, Xiaohui},
    title = "{UFold: fast and accurate RNA secondary structure prediction with deep learning}",
    journal = {Nucleic Acids Research},
    volume = {50},
    number = {3},
    pages = {e14-e14},
    year = {2021},
    month = {11},
    issn = {0305-1048},
    doi = {10.1093/nar/gkab1074},
    url = {https://doi.org/10.1093/nar/gkab1074},
}
```
