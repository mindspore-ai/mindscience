[ENGLISH](README.md)|简体中文

# DeepFRI

## 介绍

本项目使用MindSpore框架对DeepFRI模型进行构建。
DeepFRI是一种图形卷积网络，通过利用从蛋白质语言模型和蛋白质结构中提取的序列特征来预测蛋白质功能。
它可以对蛋白质进行四个方面的预测：
分子功能（Molecular Function, MF）、细胞组分（Cellular Component, CC）、生物过程（Biological Process, BP）、EC编号（Enzyme Commission, EC）。

* MF、CC、BP是Gene Ontology（基因本体论）的三大独立的本体论词汇表。GO是一个国际标准化的基因功能分类体系，
  提供了一套动态并可控的词汇表来全面描述生物体中基因和基因产物的属性，它由一组预先定义好的GO术语（GO term）组成，这组术语对基因产物的功能进行限定和描述。
  GO terms是对基因的产物而不是基因本身进行描述，因为基因的产物有时候不止一种，而GO name则是该GO term的具体名称。
  DeepFRI将输出MF、CC、BP对应的GO term与GO name。

* EC编号或EC号是酶学委员会（Enzyme Commission）为酶所制作的一套编号分类法，是以每种酶所催化的化学反应为分类基础。
  这套分类法亦同时会为各种酶给予一个建议的名称，所以亦称为酶学委员会命名法。
  针对EC，DeepFRI将直接输出其EC编号。

<div align=center>
<img src="../../../docs/deepfri_pipeline.png" alt="DeepFRI流程图" width="600"/>
</div>

参考论文地址：[DeepFRI-Paper](https://www.nature.com/articles/s41467-021-23303-9)

论文 github：[DeepFRI-Github](https://github.com/flatironinstitute/DeepFRI)

## 环境

本项目推荐使用环境版本为：

* mindspore 2.0.0

## 代码架构

```bash
├── DeepFRI
    ├── predict.py                                            // DeepFRI推理文件
    ├── train.py                                              // DeepFRI训练文件
    ├── utils.py                                              // DeepFRI功能支持文件
    ├── requirements.txt                                      // DeepFRI环境要求
    ├── README_CN.md                                          // DeepFRI相关中文说明
    ├── README.md                                             // DeepFRI相关英文说明
    ├── config
        ├── DeepFRI_cellular_component_model_params.json      // 模型 cc 参数配置
        ├── DeepFRI_enzyme_commission_model_params.json       // 模型 ec 参数配置
        ├── DeepFRI_molecular_function_model_params.json      // 模型 mf 参数配置
        ├── DeepFRI_biological_process_model_params.json      // 模型 bp 参数配置
        ├── model_config.json                                 // 四种模型总体参数配置
    ├── examples                                              // 输入样例
    ├── model
        ├── deepfri.py                                        // DeepFRI主模型文件
        ├── predictor.py                                      // DeepFRI预测支持文件
    ├── module
        ├── layers.py                                         // DeepFRI模型各层支持
    ├── trained_models                                        // 四种预训练模型存放文件夹
    ├── output                                                // 模型输出以及训练权重存放文件夹
        ├── checkpoints                                       // 训练权重存放文件夹
    ├── scripts                                               // 模型脚本存放文件夹
        ├── pred_cc.sh                                        // 测试 cc 模型脚本（PDB）
        ├── pred_mf.sh                                        // 测试 mf 模型脚本（PDB）
        ├── pred_bp.sh                                        // 测试 bp 模型脚本（PDB）
        ├── pretrained_ckpt_train.sh                          // 训练 mf 预训练模型脚本（SWISS-MODEL）
        ├── train_cc.sh                                       // 训练 cc 模型脚本（SWISS-MODEL）
        ├── train_mf.sh                                       // 训练 mf 模型脚本（SWISS-MODEL）
        ├── train_bp.sh                                       // 训练 bp 模型脚本（SWISS-MODEL）
```

## 运行：

如果您想直接进行对DeepFRI进行运行操作，您可以在本文末下载样例 `examples.zip` 与所需要的训练模型。
请在主目录下新建 `trained_models` 文件夹用以存放训练模型并将 `examples.zip` 解压。
进行运行操作时，如果您没有指定输出保存路径DeepFRI会首先在主目录下创建 `output` 文件夹用以存放结果。
同时也会在 `output` 文件夹下面创建 `checkpoints` 文件夹用以保存训练中的各个模型。

```bash
用法：python predict.py --cmap ./examples/pdb_cmaps/1S3P-A.npz -ont mf --verbose

选项：
--cmap               输入指定Protein contact map (*npz文件) 或者 protein PDB file (*pdb文件)
--npz_dir            输入包含蛋白质contact map的*npz文件目录
--pdb_dir            输入包含预测Rosetta/DMPFold结构的PDB文件目录
--save_path          输出的结果存放的文件夹路径(默认为'./output')
--ontology           用于切换不同任务(mf, ec, cc, bp)
--verbose            是否显示预测结果(action="store_true")
--evaluation_path    在验证集上计算各个阈值下的精度与召回率
--device_target      指定运行平台(Ascend, GPU, CPU)
--device_id          指定运行设备(默认: 0)
```

如果需要更换为自己训练的模型，您可以通过更改 `./config/model_config.json` 或直接替换 `./trained_models` 中的模型来实现。

如果您选择使用我们所提供的模型，DeepFRI的部分运行操作与输出结果如下所示。

### 运行操作 1: 从蛋白质 contact map 预测蛋白质的功能

样例: 利用 Parvalbumin alpha 蛋白的序列和 contact map 预测其MF-GO项 (PDB: [1S3P](https://www.rcsb.org/structure/1S3P)):

```bash
>> python predict.py --cmap ./examples/pdb_cmaps/1S3P-A.npz -ont mf --verbose

```

### 输出

```txt
Protein GO-term/EC-number Score GO-term/EC-number name
query_prot GO:0005509 0.99995 calcium ion binding
```

### 运行操作 2: 从 contact map 目录预测蛋白质的功能

```bash
>> python predict.py --npz_dir examples/pdb_cmaps -ont mf -v

```

### 输出

```txt
Protein GO-term/EC-number Score GO-term/EC-number name
pdb_cmaps\1S3P-A GO:0005509 0.99995 calcium ion binding
pdb_cmaps\2J9H-A GO:0004364 0.97407 glutathione transferase activity
pdb_cmaps\2J9H-A GO:0016765 0.88968 transferase activity, transferring alkyl or aryl (other than methyl) groups
pdb_cmaps\2J9H-A GO:0042277 0.40748 peptide binding
...
pdb_cmaps\2W83-E GO:0016818 0.83582 hydrolase activity, acting on acid anhydrides, in phosphorus-containing anhydrides
pdb_cmaps\2W83-E GO:0016817 0.83478 hydrolase activity, acting on acid anhydrides
pdb_cmaps\2W83-E GO:0003924 0.80225 GTPase activity
pdb_cmaps\2W83-E GO:0019899 0.13060 enzyme binding
```

### 运行操作 3: 从 PDB 文件目录中预测蛋白质的功能

```bash
>> python predict.py --pdb_dir ./examples/pdb_files -ont mf

```

### 输出

您可以看见文件保存在: `./output/`

## 训练

对DeepFRI进行训练时，如果您不打算使用我们提供的预训练模型，请您首先下载 **`DeepFRI_LSTM.ckpt`** 文件。
DeepFRI在进行训练时，会固定LSTM层的参数。感谢 [DeepFRI-Paper](https://www.nature.com/articles/s41467-021-23303-9) 的工作，
LSTM层参数迁移自 [Newest Models](https://users.flatironinstitute.org/vgligorijevic/public_www/DeepFRI_data/newest_trained_models.tar.gz) 中的lstm_lm_tf.hdf5模型
（其余各模型的LSTM层参数与lstm_lm_tf.hdf5参数同样为一致的）

```bash
用法：python train.py -device "Ascend" -id 0 -ont mf -out "./output_mf" \
--pretrained_ckpt_path "./trained_models/DeepFRI_molecular_function.ckpt"

选项：
--epochs                 训练总轮数(训练中设有早停)
--ontology               选择训练模型的类别(mf, ec, cc, bp)
--device_target          指定训练平台(Ascend, GPU, CPU)
--device_id              指定训练设备(默认: 0)
--output_dir             指定checkpoints输出路径(默认新建'./output/checkpoints/'文件夹并储存)
--pretrained_ckpt_path   指定预训练模型(预训练模型LSTM层参数训练中固定)
```

如果需要对各模型的参数与训练参数进行更改，请直接更改各 `./config/..._model_params.json` 文件。
同时您可以运行 `./scripts/train_xx.sh` 从头开始对模型训练，或者使用 `pretrained_ckpt_train.sh` 对预训练模型进行训练。
对于验证，请您运行 `./scripts/pred_xx.sh` 以测得已训练模型各个阈值下的精度与召回率。

对于 `./config/..._model_params.json` 文件，部分特殊参数说明如下：

* gc_dims: MultiGraphConv三层维度设置，默认[512, 512, 512]
* fc_dims: 全连接层维度设置
* pad_len: 对cmap进行补齐操作，默认1024
* goterms: GO-term/EC-number
* gonames: GO-name/EC-number
* cmap_type: Contact maps类型，可选['ca', 'cb']，默认ca
* cmap_thresh: 阈值，默认10.0

*注：本项目由于**SWISS-MODEL-EC.tar.gz**数据集链接失效，故没有训练ec模型，但保留ec的预测与训练功能。*

## 测试样例、数据集、模型

训练所用数据集分别为从PDB数据库和SWISS-MODEL数据库中挑选的条目构建的集合。
作者选取带有注释的PDB链与SWISS-MODEL链，删除相同和相似的序列，
通过在95%序列同一性（即序列比对中残基总数中相同残基的数量）下对所有PDB链和SWISS-MODEL链（能够检索到contact map）进行聚类来创建非冗余集。

* PDB蛋白质结构数据库(Protein Data Bank,简称PDB)是美国Brookhaven国家实验室于1971年创建的，由结构生物信息学研究合作组织(Research Collaboratory for Structural Bioinformatics,简称RCSB)维护。

* SWISS-MODEL知识库是一个蛋白质3D结构数据库，库中收录的蛋白质结构都是使用SWISS-MODEL同源建模方法（homology-modelling）得来的。

作者所设计的数据集(训练和验证)作为tensorflow特定的“TFRecord”文件提供，它们可以从以下网站下载。

| 所属模块   | 文件名        | 大小 | 描述  |Model URL  |
|-----------|---------------------|---------|---------------|-----------------------------------------------------------------------|
| examples样例 | `examples.zip` | 1.7MB | 测试所用的样例，包含了PDB文件、npz文件等 |  [下载链接](https://download.mindspore.cn/mindscience/mindsponge/DeepFRI/examples/examples.zip) |
| PDB数据集 | `PDB-GO.tar.gz` | 19GB | PDB-GO数据集，可用于mf、bp、cc功能的训练 | [下载链接](https://users.flatironinstitute.org/vgligorijevic/public_www/DeepFRI_data/PDB-GO.tar.gz) |
| PDB数据集 | `PDB-EC.tar.gz` | 13GB | PDB-EC数据集，可用于ec功能的训练 |  [下载链接](https://users.flatironinstitute.org/vgligorijevic/public_www/DeepFRI_data/PDB-EC.tar.gz) |
| SWISS-MODEL数据集 | `SWISS-MODEL-GO.tar.gz` | 165GB | SWISS-MODEL数据集，可用于mf、bp、cc功能的训练 |  [下载链接](https://users.flatironinstitute.org/vgligorijevic/public_www/DeepFRI_data/SWISS-MODEL-GO.tar.gz) |
| SWISS-MODEL数据集 | `SWISS-MODEL-EC.tar.gz` | 117GB | SWISS-MODEL数据集，可用于ec功能的训练 |  [下载链接](https://users.flatironinstitute.org/vgligorijevic/public_www/DeepFRI_data/SWISS-MODEL-EC.tar.gz) |
| BP模型 | `DeepFRI_biological_process.ckpt` | 74.5MB | DeepFRI针对BP任务在PDB数据集与SWISS-MODEL数据集联合训练的checkpoint链接 | [下载链接](https://download.mindspore.cn/mindscience/mindsponge/DeepFRI/checkpoint/DeepFRI_biological_process.ckpt) |
| CC模型 | `DeepFRI_cellular_component.ckpt` | 40.6MB | DeepFRI针对BP任务在PDB数据集与SWISS-MODEL数据集联合训练的checkpoint链接 | [下载链接](https://download.mindspore.cn/mindscience/mindsponge/DeepFRI/checkpoint/DeepFRI_cellular_component.ckpt) |
| MF模型 | `DeepFRI_molecular_function.ckpt` | 42.0MB | DeepFRI针对BP任务在PDB数据集与SWISS-MODEL数据集联合训练的checkpoint链接 | [下载链接](https://download.mindspore.cn/mindscience/mindsponge/DeepFRI/checkpoint/DeepFRI_molecular_function.ckpt) |
| LSTM层权重 | `DeepFRI_LSTM.ckpt` | 20.0MB | DeepFRI使用的LSTM层权重链接 | [下载链接](https://download.mindspore.cn/mindscience/mindsponge/DeepFRI/checkpoint/DeepFRI_LSTM.ckpt) |
