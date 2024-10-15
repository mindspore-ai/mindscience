
# 模型名称

> DiffCSP

## 介绍

> DiffCSP是基于图神经网络和等变扩散模型的晶体生成模型，功能：给定组分，预测晶体材料的结构。

## 环境要求

> 1. 安装`mindspore（2.3.0）`
> 2. 安装依赖包：`pip install -r requirement.txt`

## 快速入门

> 1. 将Mindchemistry/mindchemistry文件包下载到当前目录
> 2. 在[数据集链接](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/)下载相应的数据集
> 3. 安装依赖包：`pip install -r requirement.txt`
> 4. 训练命令： `python train.py`
> 5. 预测命令： `python evaluate.py`
> 6. 评估命令： `python compute_metric.py`
> 7. 评估结果放在`config.yaml`中指定的`metric_dir`路径的json文件中

### 代码目录结构

```txt
diffcsp
    │  README.md    README文件
    │  config.yaml    配置文件
    │  train.py     训练启动脚本
    │  evaluate.py     推理启动脚本
    │  compute_metric.py     评估启动脚本
    │  requirement.txt    环境依赖
    │  
    └─data
            data_utils.py  数据集处理工具
            dataset.py 读取数据集
            crysloader.py 数据集载入器
    └─models
            cspnet.py  基于图神经网络的去噪器模块
            diffusion.py   扩散模型模块
            diff_utils.py  工具模块
            infer_utils.py  推理工具模块
            train_utils.py  训练工具模块

```

## 下载数据集

在[数据集链接](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/)中下载相应的数据集文件夹和dataset_prop.txt数据集属性文件放置于当前路径的dataset文件夹下（如果没有需要自己手动创建），文件路径参考：

```txt
diffcsp
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

将权重的path写入config文件的checkpoint.last_path中。预训练模型可以从[预训练模型链接](https://download-mindspore.osinfra.cn/mindscience/mindchemistry/diffcsp/pre-train)中获取。

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
{"match_rate": 0.985997357992074, "rms_dist": 0.013073775170360118}
```
