## 目录

- [目录](#目录)
- [点云散射参数预测](#点云散射参数预测)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
- [随机情况说明](#随机情况说明)
- [MindScience主页](#mindscience主页)

## 点云散射参数预测

传统电磁仿真计算方法基于有限元或有限差分方法，计算过程中需要进行大量迭代计算，导致计算时间长，无法满足现代产品设计对仿真效率的需求。AI方法通过端到端仿真计算直接计算目标结构散射参数，可以大幅提升仿真速度，缩短产品的研发周期。

## 数据集

基于[点云数据生成](https://gitee.com/mindspore/mindscience/tree/master/MindElec/examples/data_driven/pointcloud/generate_pointcloud)和[点云数据压缩](https://gitee.com/mindspore/mindscience/tree/master/MindElec/examples/data_driven/pointcloud/data_compression)生成散射参数预测输入数据。而标签数据需要使用商业仿真软件或者时域有限差分算法生成。

## 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindElec](https://gitee.com/mindspore/mindscience/tree/master/MindElec)
- 如需查看详情，请参见如下资源：
    - [MindELec教程](https://www.mindspore.cn/mindscience/docs/zh-CN/master/mindelec/intro_and_install.html)
    - [MindElec Python API](https://www.mindspore.cn/mindscience/api/zh-CN/master/mindelec.html)

## 脚本说明

### 脚本及样例代码

```path
.
└─S_parameter
  ├─README.md
  ├─src
    ├─config.py                      ## 超参定义
    ├─dataset.py                     ## 数据集准备与导入
    ├─metric.py                      ## 评估指标
    ├─model.py                       ## 网络模型
    ├─lr_generator.py                ## 学习率生成
  ├──train.py                        ## 训练网络
  ├──eval.py                         ## 评估网络
```

### 脚本参数

在`config.py`中可以同时配置训练参数。

```python
'input_channels': 4,                      ## 输入张量的特征数
'epochs': 4000,                           ## 训练轮次
'batch_size': 8,                          ## 批数据大小
'lr': 0.0001,                             ## 基础学习率
'lr_decay_milestones': 5,                 ## LR衰减次数
```

### 训练过程

您可以通过train.py脚本训练压缩模型，训练结束后模型参数会自动保存：

``` shell
python train.py --train_input_path TRAIN_INPUT_PATH
                --train_label_path TRAIN_LABEL_PATH
                --data_config_path DATA_CONFIG_PATH
                --device_num 0
                --checkpoint_dir CKPT_PATH
```

### 评估过程

模型训练结束后可以通过eval.py开始评估过程：

``` shell
python eval.py  --input_path TEST_INPUT_PATH
                --label_path TEST_LABEL_PATH
                --data_config_path DATA_CONFIG_PATH
                --device_num 0
                --model_path CKPT_PATH
                --output_path OUTPUT_PATH
```

评估过程中会实时绘制预测值与真时值的对比图，并保存在输出目录下。

## 随机情况说明

train.py中设置了“create_dataset”函数内的种子。

## MindScience主页

请浏览官网[主页](https://gitee.com/mindspore/mindscience)。
