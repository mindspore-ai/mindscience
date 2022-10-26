# DeepDR

## 介绍

DeepDR是一种基于网络的深度学习方法，通过整合十个网络进行硅药物再利用：一个药物-疾病，一个药物副作用，一个药物靶点和7个药物-药物网络，由此揭示药物和疾病之间的潜在关联。deepDR首先融合不同网络类型的信息到一个紧凑的，低维特征表示，然后学习药物特性的低维表示一起与已知的药物疾病交互对输入变分自编码器预测新的药物疾病。具体来讲，DeepDR通过多模态深度自编码器(MDA)从异构网络中学习药物的高级特征。此外，通过应用多层非线性函数来保持非线性网络结构。它还用药物特征补充了稀疏评级，因为将侧信息输入相同的VAE增加了用于训练的样本数量。具体来说，药物-疾病关联和药物特征是不同的信息来源，都是药物的信息，因此可以通过相同的推理网络和生成网络进行集体编码和解码。然后，通过变分自动编码器(cVAE)对临床报告的药物对-疾病对进行集体编码和解码，以推断出最初未被批准的已批准药物的候选药物。DeepDR显示出高性能，优于传统的基于网络或基于机器学习的方法。

Multi-modal deep autoencoder：
![multi-modal deep autoencoder](multi-modal%20deep%20autoencoder.png)

Collective variational autoencoder：
![collective variational autoencoder](collective%20variational%20autoencoder.png)

## 环境

本项目运行于Nvidia RTX3090，采用Mindspore深度学习框架。本项目可以通过自行配置运行环境使其部署于其他硬件环境。

本项目使用环境版本为：

mindspore-gpu 1.8.0；

python 3.7；

## 组织架构

- src：数据预处理和模型脚本；
- train_mda：MDA模型训练脚本；
- train_cvae：cVAE模型训练脚本；
- eval：MDA模型和cVAE模型推理脚本。

## 数据集和模型参数文件

本项目提供数据集和已训练模型参数的checkpoint文件，存放于[此链接](https://gitee.com/bling__bling/deep-dr.git)，供开发人员下载使用。
其中，数据集包括：

1. 两个常用数据库DrugBank和repoDB的数据组合而成的drug–disease网络：(i) clinically reported drug–drug interactions, (ii) drug–target interactions, (iii) drug-side-effect associations, (iv) chemical similarities, (v) therapeutic similarities derived from the Anatomical Therapeutic Chemical Classification System, (vi) drugs’ target sequence similarities, (vii) Gene Ontology (GO) biological process, (viii) GO cellular component and (ix) GO molecular function. 它们存放于datasets/drugNets目录下。
2. 临床报告的6677个drug–disease对，连接1519种药物和1229种疾病，用于构建预测深度学习模型。存放于datasets/drugDisease.txt目录下。
3. 经数据预处理脚本处理得到的PPMI矩阵。存放于datasets/PPMI目录下。

## conda环境配置

```conda环境配置
- conda create --name deepdr
- conda activate deepdr
- conda install mindspore-gpu=1.8.1 cudatoolkit=11.1 -c mindspore -c conda-forge
```

## 运行

DeepDR模型训练推理流程主要包括：（1）deepDR通过多模态深度自编码器(MDA)将每个网络的PPMI矩阵融合成一个紧凑的、所有网络通用的低维特征表示，然后从MDA的中间层提取低维特征；（2）DeepDR使用cVAE来推断药物和疾病之间的潜在联系。具体来说，它将从第二步提取的高质量特征输入到VAE进行预训练，然后通过输入药物-疾病关联网络对VAE进行微调。

MDA模型训练脚本运行示例。

```text
- cd  /root/DeepDR
- python train_mda.py --epochs 150 --batch_size 64 --noise_factor 0.5
```

MDA模型推理脚本运行示例。

```text
- cd  /root/DeepDR
- python eval.py --model 'mda'
```

cVAE模型训练脚本运行示例。
预训练：

```text
- cd  /root/DeepDR
- python train_cvae.py --a 6 --b 0.1 --m 300 --batch 100 --save --learn_rate 0.001
```

微调：

```text
- cd  /root/DeepDR
- python train_cvae.py --a 15 --b 3 --load 1 --m 300 --batch 100 --rating --save --learn_rate 0.001
```

cVAE推理脚本运行示例。

```text
- cd  /root/DeepDR
- python eval.py --model 'cvae'
```
