# UFold

## 模型介绍

对于许多RNA分子来说，二级结构对于RNA的正确功能至关重要。从核苷酸序列预测RNA二级结构是基因组学中一个长期存在的问题，但随着时间的推移，预测性能已经达到了稳定水平。传统的RNA二级结构预测算法主要基于热力学模型，通过自由能最小化，这强加了很强的先验假设，而且运行速度很慢。UFold作为一种基于深度学习的方法，用于RNA二级结构预测，直接根据注释数据和碱基配对规则进行训练。UFold提出了一种新的RNA序列的类图像表示方法，它可以通过完全卷积网络(FCNs)进行有效的处理。

模型的输入是通过取One-Hot Encoding的四个基本通道的所有组合的外积生成的，这产生了16个通道。然后，表示配对概率的附加信道与16信道序列表示串联，并一起作为模型的输入。UFold模型是U-Net的一个变体，它将17通道张量作为输入，并通过连续卷积和最大池运算转换数据。

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
