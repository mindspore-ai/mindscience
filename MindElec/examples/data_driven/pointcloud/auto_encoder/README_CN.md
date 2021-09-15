# 目录

- [目录](#目录)
- [点云数据压缩](#点云数据压缩)
  - [压缩模型自监督训练](#压缩模型自监督训练)
  - [数据压缩](#数据压缩)
- [数据集](#数据集)
- [特性](#特性)
  - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
  - [脚本及样例代码](#脚本及样例代码)
  - [脚本参数](#脚本参数)
  - [压缩模型训练](#压缩模型训练)
    - [用法](#用法)
  - [数据分块压缩](#数据分块压缩)
    - [用法](#用法-1)
- [随机情况说明](#随机情况说明)
- [MindScience主页](#mindscience主页)

# 点云数据压缩

使用点云数据计算散射参数时，如果目标体系结构复杂或结构调整比较精细，点云数据的分辨率需要设置得非常高才能保证点云数据的有效性；但同时，分辨率高会导致单条数据过大的问题，以手机仿真为例，该场景中单条点云数据通常包含上亿个点，深度仿真方法处理这种数据需要大量的显存与算力，通用醒与高效性也会降低。

针对该问题我们提出使用基于神经网络的压缩模型对原始点云数据做分块压缩，该压缩流程分为压缩模型自监督训练与数据压缩两步：

## 压缩模型自监督训练

- 随机从点云数据中取出25 * 50 * 25的数据块作为数据集
- 构建基于AutoEncoder结构的压缩-重建模型，以最小化重建误差为目标训练该模型
- 保存Encoder部分checkpoint

## 数据压缩

- 将点云数据分为25 * 50 * 25的块数据，分批压缩
- 将压缩向量按分块原空间位置排布，形成压缩数据


# 数据集

使用src/dataset.py中的generate_data函数可以自动从原始点云数据中随机取出25 * 50 * 25 的块数据用于训练或者测试

# 特性

## 混合精度

默认采用[混合精度](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_mixed_precision.html)的训练方法提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。

以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend或GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)


# 脚本说明

## 脚本及样例代码

```path
.
└─auto_encoder
  ├─README.md
  ├─src
    ├─dataset.py                     # 数据集准备与导入
    ├─metric.py                      # 评估指标
    ├─model.py                       # 网络模型
    ├─lr_generator.py                # 学习率生成
  ├──train.py                        # 自监督训练压缩模型
  ├──data_compress.py                # 数据分块压缩
```

## 脚本参数

在train.py中可以配置训练参数。

```python
'base_channels': 8,                       # 压缩模型特征基数
'input_channels': 4,                      # 输入张量的特征数
'epochs': 2000,                           # 训练轮次
'batch_size': 128,                        # 批数据大小
'save_epoch': 100,                        # 检查点保存间隔
'lr': 0.001,                              # 基础学习率
'lr_decay_milestones': 5,                 # LR衰减次数
'eval_interval': 20,                      # 评估间隔
'patch_shape': [25, 50, 25],              # 块数据大小
```

## 压缩模型训练

### 用法

您可以通过train.py脚本训练压缩模型，训练过程中Encoder部分参数会自动保存：
```
python train.py --train_input_path TRAIN_INPUT_PATH 
                --test_input_path TEST_INPUT_PATH 
                --device_num 0 
                --checkpoint_dir CKPT_PATH
```

## 数据分块压缩

### 用法

压缩模型训练结束后可以通过data_compress.py开始数据分块压缩：
```
python data_compress.py --input_path TEST_INPUT_PATH 
                        --data_config_path DATA_CONFIG_PATH
                        --device_num 0 
                        --model_path CKPT_PATH
                        --output_save_path OUTPUT_PATH
```

压缩结束您可以在压缩文件输出目录中找到压缩后的点云数据。


# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还设置了“generate_data”随机划分训练测试集中的随机种子。

# MindScience主页
请浏览官网[主页](https://gitee.com/mindspore/mindsciencetmp)。
