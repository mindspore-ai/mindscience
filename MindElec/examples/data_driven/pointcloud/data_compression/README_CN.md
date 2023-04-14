# 目录

- [目录](#目录)
- [点云数据压缩](#点云数据压缩)
    - [压缩模型自监督训练](#压缩模型自监督训练)
    - [数据分块压缩](#数据分块压缩)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [压缩模型训练](#压缩模型训练)
        - [用法](#用法)
    - [数据分块压缩](#数据分块压缩-1)
        - [用法](#用法-1)
- [随机情况说明](#随机情况说明)
- [MindScience主页](#mindscience主页)

# 点云数据压缩

使用点云数据计算散射参数时，如果目标体系结构复杂或结构调整比较精细，点云数据的分辨率需要设置得非常高才能保证点云数据的有效性；但同时，分辨率高会导致单条数据过大的问题，以手机仿真计算为例，该场景中单条点云数据通常包含上亿个点，深度仿真计算方法处理这种数据需要大量的显存与算力，通用性与高效性也会降低。

针对该问题MindSpore Elec提供基于神经网络的分块压缩工具，该工具可以大幅降低点云数据方案的显存与算力消耗，提升基于点云数据AI仿真方案的通用性与高效性。

## 压缩模型自监督训练

- 随机从点云数据中取出 25 x 50 x 25 的数据块作为训练数据集。
- 构建基于AutoEncoder结构的压缩-重建模型，以最小化重建误差为目标训练该模型。
- 保存Encoder部分checkpoint。

## 数据分块压缩

- 将点云数据分为 25 x 50 x 25 的块数据，分块压缩。
- 将压缩向量按分块前的原空间位置排布，形成压缩数据。

# 数据集

使用src/dataset.py中的generate_data函数可以自动从原始点云数据中随机取出 25 x 50 x 25 的块数据用于训练或者测试。

本示例中模型训练数据涉及商业机密，无法提供下载地址，开发者可以使用`src/sampling.py`生成用于功能验证的伪数据。

# 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore Elec](https://gitee.com/mindspore/mindscience/tree/master/MindElec)
- 如需查看详情，请参见如下资源：
    - [MindSpore Elec教程](https://www.mindspore.cn/mindelec/docs/zh-CN/master/intro_and_install.html)
    - [MindSpore Elec Python API](https://www.mindspore.cn/mindelec/docs/zh-CN/master/mindelec.architecture.html)

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
    ├─sampling.py                    # 伪数据生成
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

``` shell
python train.py --train_input_path TRAIN_INPUT_PATH
                --test_input_path TEST_INPUT_PATH
                --device_num 0
                --checkpoint_dir CKPT_PATH
```

## 数据分块压缩

### 用法

压缩模型训练结束后可以通过`data_compress.py`开始数据分块压缩：

``` shell
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

请浏览官网[主页](https://gitee.com/mindspore/mindscience)。
