# 目录

- [目录](#目录)
- [点云数据电磁仿真](#点云数据电磁仿真)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过错](#训练过程)
    - [推理过程](#推理过程)
- [随机情况说明](#随机情况说明)
- [MindScience主页](#mindscience主页)

# 点云数据电磁仿真

传统电磁仿真计算方法基于有限元或有限差分方法，计算过程中需要进行大量迭代计算，导致计算时间长，无法满足现代产品设计对仿真效率的需求。AI方法通过端到端仿真计算跳过迭代计算直接得出电磁场分布，可以大幅提升仿真速度，缩短产品的研发周期。

# 数据集

基于[点云数据生成](https://gitee.com/mindspore/mindscience/tree/master/MindElec/examples/data_driven/pointcloud/generate_pointcloud)和[点云数据压缩](https://gitee.com/mindspore/mindscience/tree/master/MindElec/examples/data_driven/pointcloud/data_compression)生成电磁场输入数据，再调用`src/inputs_process.py`将输入数据绑定坐标和源位置信息。

标签数据需要使用商业仿真软件或者时域有限差分算法生成，再调用`src/label_process.py`处理得到最终的标签。

我们训练过程中使用的手机数据涉及商业机密，所以无法对外展示，可以使用`src/sample.py`脚本生成随机数据用于模型功能验证。

# 环境要求

- 硬件（Ascend）
    - 准备Ascend处理器搭建硬件环境。
- 框架
    - [MindSpore Elec](https://gitee.com/mindspore/mindscience/tree/master/MindElec)
- 如需查看详情，请参见如下资源：
    - [MindSpore Elec教程](https://www.mindspore.cn/mindelec/docs/zh-CN/master/intro_and_install.html)
    - [MindSpore Elec Python API](https://www.mindspore.cn/mindelec/docs/zh-CN/master/mindelec.architecture.html)

# 快速入门

通过官方网站安装MindSpore Elec后，您可以按照如下步骤进行训练和验证：

- Ascend处理器环境运行

```python
# 训练运行示例
bash run_train.sh [DATASET_PATH]

# 推理运行示例
bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

# 脚本说明

## 脚本及样例代码

```path
.
└─full_em
    ├─README.md
    ├─scripts
        ├──run_train.sh                   # 启动Ascend训练
        ├──run_eval.sh                    # 启动Ascend推理
    ├─src
        ├──inputs_process.py              # 训练输入前处理
        ├──label_process.py               # 训练标签前处理
        ├──config.py                      # 参数配置
        ├──dataset.py                     # 数据集配置
        ├──loss.py                        # 损失函数
        ├──maxwell_model.py               # 点云数据电磁仿真模型
    ├──train.py                           # 训练网络
    ├──eval.py                            # 评估网络
```

## 脚本参数

在`src/config.py`中可以配置训练参数。

```python
"epochs": 500,                            # 训练轮次
"batch_size": 8,                          # 批数据大小
"lr": 0.0001,                             # 基础学习率
"t_solution": 162,                        # 时间轴分辨率
"x_solution": 50,                         # 空间X轴分辨率
"y_solution": 50,                         # 空间Y轴分辨率
"z_solution": 8,                          # 空间Z轴分辨率
"save_checkpoint_epochs": 5,              # 检查点保存间隔
"keep_checkpoint_max": 20                 # 最大保存检查点数
```

## 训练过程

```shell
  bash run_train.sh  [DATASET_PATH]
```

此脚本需设置以下参数：

- `DATASET_PATH`：训练数据集的路径

训练结果保存在当前路径下“train”文件夹中。您可在日志中找到checkpoint文件以及结果。

## 推理过程

在运行以下命令之前，请检查用于推理的checkpoint路径。请将checkpoint路径设置为绝对路径，如`username/train/ckpt/Maxwell3d-5_1275.ckpt`。

```shell
  bash run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH]
```

此脚本需设置两个参数：

- `DATASET_PATH`：验证数据集的路径。
- `CHECKPOINT_PATH`：checkpoint文件的绝对路径。

> 训练过程中可以生成checkpoint。

推理结果保存在示例路径，文件夹名为`eval`。您可在日志中找到如下结果。

```shell
  test_res:  l2_error: 0.9250676897
```

# 随机情况说明

dataset.py中设置了“create_dataset”函数内的种子，同时还使用了train.py中的随机种子。

# MindScience主页

请浏览官网[主页](https://gitee.com/mindspore/mindscience)。
