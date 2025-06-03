
# 模型名称

> CrystalFlow

## 介绍

> 理论晶体结构预测是通过计算的手段寻找物质在给定的外界条件下最稳定结构的重要手段。传统结构预测方法依赖在势能面上广泛的随机采样来寻找最稳定结构，然而，这种方法需要对大量随机生成的结构进行局域优化，而局域优化通常需要消耗巨大的第一性原理计算成本，尤其在模拟多元素复杂体系时，这种计算开销会显著增加，从而带来巨大的挑战。近年来，基于深度学习生成模型的晶体结构生成方法因其能够在势能面上更高效地采样合理结构而逐渐受到关注。这种方法通过从已有的稳定或局域稳定结构数据中学习，进而生成合理的晶体结构，与随机采样相比，不仅能够减少局域优化的计算成本，还能通过较少的采样找到体系的最稳定结构。采用神经常微分方程和连续变化建模概率密度的归一化流流模型，相比采用扩散模型方法的生成模型具有更加简洁、灵活、高效的优点。本方法基于流模型架构，发展了以CrystalFlow命名的晶体结构生成模型，在MP20等基准数据集上达到优秀的水平。

## 环境要求

> 1. 安装`mindspore（2.5.0）`
> 2. 安装依赖包：`pip install -r requirement.txt`

## 快速入门

> 1. 将Mindchemistry/mindchemistry文件包下载到当前目录
> 2. 在[数据集链接](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/dataset/)下载相应的数据集
> 3. 安装依赖包：`pip install -r requirement.txt`
> 4. 训练命令： `python train.py`
> 5. 预测命令： `python evaluate.py`
> 6. 评估命令： `python compute_metric.py`
> 7. 评估结果放在`config.yaml`中指定的`metric_dir`路径的json文件中

### 代码目录结构

```text
代码主要模块在models文件夹下，其中cspnet.py是网络层，flow.py是流模型模块.data文件夹下是数据集处理模块。

applications
  └── crystalflow                                      # 模型名
        ├── readme.md                                  # readme文件
        ├── config.yaml                                # 配置文件
        ├── train.py                                   # 训练启动脚本
        ├── evaluate.py                                # 推理启动脚本
        ├── compute_metric.py                          # 评估启动脚本
        ├── requirement.txt                             # 环境依赖
        ├── data                                       # 数据处理模块
        |     ├── data_utils.py                        # 工具模块
        |     ├── dataset.py                           # 构造数据集
        |     └── crysloader.py                        # 构造数据加载器
        └── models
              ├── conditioning.py                      # 条件生成工具模块
              ├── cspnet.py                            # 基于图神经网络的去噪器模块
              ├── cspnet_condition.py                  # 条件生成的网络层
              ├── diff_utils.py                        # 工具模块
              ├── flow.py                              # 流模型模块
              ├── flow_condition.py                    # 条件生成的流模型
              ├── infer_utils.py                       # 推理工具模块
              ├── lattice.py                           # 晶格矩阵处理工具
              └── train_utils.py                       # 训练工具模块

```  

## 下载数据集

在[数据集链接](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/dataset/)中下载相应的数据集文件夹和dataset_prop.txt数据集属性文件放置于当前路径的dataset文件夹下（如果没有需要自己手动创建），文件路径参考：

```txt
crystalflow
    ...
    └─dataset
            perov_5 钙钛矿数据集
            carbon_24 碳晶体数据集
            mp_20 晶胞内原子数最多为20的MP数据集
            mpts_52 晶胞内原子数最多为52的MP数据集
            dataset_prop.txt 数据集属性文件
    ...
```

## 训练过程

### 训练

将Mindchemistry/mindchemistry文件包下载到当前目录;

更改config文件，设置训练参数:
> 1. 设置训练的dataset，见dataset字段
> 2. 设置去噪器模型的配置，见model字段
> 3. 设置训练保存的权重文件，更改train.ckpt_dir文件夹名称和checkpoint.last_path权重文件名称
> 4. 其它训练设置见train字段

```bash
pip install -r requirement.txt
python train.py
```

### 推理

更改config文件中的test字段来更改推理参数，特别是test.num_eval，它**决定了对于每个组分生成多少个样本**，对于后续的评估阶段很重要。

```bash
python evaluate.py
```

推理得到的晶体将保存在test.eval_save_path指定的文件中

文件中存储的内容为python字典，格式为：

```python
{
        'pred': [
                [晶体A sample 1, 晶体A sample 2, 晶体A sample 3, ... 晶体A sample num_eval],
                [晶体B sample 1, 晶体B sample 2, 晶体B sample 3, ... 晶体B sample num_eval]
                ...
        ]
        'gt': [
                晶体A ground truth,
                晶体B ground truth,
                ...
        ]
}
```

### 评估

将推理得到的晶体文件的path写入config文件的test.eval_save_path中；

确保num_evals与进行推理时设置的对于每个组分生成样本的数量一致或更小。比如进行推理时，num_evals设置为1，那么评估时，num_evals只能设置为1；推理时，num_evals设置为20，那么评估时，num_evals可以设置为1-20的数字来进行评估。

更改config文件中的test.metric_dir字段来设置评估结果的保存路径

```bash
python compute_metric.py
```

得到的评估结果文件示例：

```json
{"match_rate": 0.6107671899181959, "rms_dist": 0.07492558322002925}
```
