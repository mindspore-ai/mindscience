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

## 点云预测散射参数

传统电磁仿真计算基于有限元或有限差分方法，计算过程中需要进行大量迭代计算，导致计算时间长，无法满足现代产品设计对仿真效率的需求。AI方法通过端到端仿真计算可以跳过迭代计算直接得出目标结构散射参数，大幅提升仿真速度，缩短产品的研发周期。

某些产品设计场景中目标结构无法用固定的一组参数来描述，针对这些场景我们提出基于点云张量数据的散射参数AI计算方法，该方法可以预测任意复杂结构（手机、PCB板等）的散射参数。

## 数据集

基于[点云数据生成](https://gitee.com/mindspore/mindscience/tree/master/MindElec/examples/data_driven/pointcloud/generate_pointcloud)和[点云数据压缩](https://gitee.com/mindspore/mindscience/tree/master/MindElec/examples/data_driven/pointcloud/data_compression)生成散射参数预测模型的输入数据，对应的标签需要使用商业仿真软件或者时域有限差分算法生成。

生成的数据集需要使用src/data_preprocessing.py中的generate_data函数进行归一化与平滑处理，调用方式参见`脚本说明/数据处理`。

本示例中模型的训练数据涉及商业机密，无法提供下载地址，开发者可以使用`src/sampling.py`生成用于功能验证的伪数据。

## 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore Elec](https://gitee.com/mindspore/mindscience/tree/master/MindElec)
- 如需查看详情，请参见如下资源：
    - [MindSpore Elec教程](https://www.mindspore.cn/mindelec/docs/zh-CN/master/intro_and_install.html)
    - [MindSpore Elec Python API](https://www.mindspore.cn/mindelec/docs/zh-CN/master/mindelec.architecture.html)

## 脚本说明

### 脚本及样例代码

```path
.
└─S_parameter
  ├─README.md
  ├─src
    ├─config.py                      ## 超参定义
    ├─dataset.py                     ## 数据集导入接口
    ├─data_preprocessing.py          ## 数据预处理
    ├─metric.py                      ## 评估指标
    ├─model.py                       ## 网络模型
    ├─lr_generator.py                ## 学习率生成
    ├─sampling.py                    ## 伪数据生成
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

### 数据处理

训练或测试数据需要调用`src/data_preprocessing.py`来对数据做归一化与平滑处理，调用方式如下。

``` shell
cd src
python data_preprocessing.py --input_path INPUT_PATH
                             --label_path LABEL_PATH
```

### 训练过程

您可以通过train.py脚本训练压缩模型，训练结束后模型参数会自动保存：

``` shell
python train.py --train_input_path TRAIN_INPUT_PATH
                --train_label_path TRAIN_LABEL_PATH
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
