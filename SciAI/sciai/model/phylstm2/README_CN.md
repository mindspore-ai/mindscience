[ENGLISH](README.md) | 简体中文

# 目录

- [基于物理的双LSTM网络](#基于物理的双lstm网络)
    - [模型介绍](#模型介绍)
    - [实现案例](#实现案例)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [基于物理的双LSTM网络](#目录)

### [模型介绍](#目录)

本工作提出了一种创新的基于物理的深度学习框架，用于对数据稀少的非线性结构系统进行元建模。基本概念是将可用但不完整的物理知识（如物理定律、科学原理）纳入深度长短期记忆（LSTM）网络，从而在可行的解决方案空间内约束和促进学习。同时将物理约束嵌入到损失函数中，以加强模型训练。具体来说，对于动力结构考虑了运动方程的物理定律、状态依赖性和滞回本构关系来构造物理损失。

> [论文](https://www.sciencedirect.com/science/article/pii/S0045782520304114)：Zhang R, Liu Y, Sun H. Physics-informed multi-LSTM networks for metamodeling of nonlinear structures[J]. Computer Methods in Applied Mechanics and Engineering, 2020, 369: 113226.

### [实现案例](#目录)

论文借助两个LSTM网络，使用从数据集中加载的原始数据送入网络以得到预测输出。具体来说，将训练数据和辅助数据同时馈入网络，在不同阶段得到对应输出后应用MSE损失计算预测值与真实值之间的差距以指导网络训练。

对于基于图的张量微分器，则使用由原始数据处理得到的中间量$\phi$与对应输出相乘实现。

![PhyLSTM2 Network](docs/PhyLSTM2_Network.png)

## [数据集](#目录)

数据集采用SDOF Bouc-Wen迟滞模型。Bouc-Wen迟滞模型是一个非线性系统，如具有速率相关的迟滞（例如，依赖于$\dot r$）。

[原始数据](./data/data_boucwen.mat)由100个样本（如独立地震序列）组成，通过对随机带限白噪声（BLWN）地面运动激发的单自由度非线性系统进行数值模拟，生成不同幅度的合成数据库。所有数据集都被格式化为$PhyLSTM^2$所需的三维数组。

论文采用训练和验证的全监督方式训练，数据集在运行过程中实时生成，训练与测试数据[生成方式](./src/process.py)如下：

- 训练数据：随机选取10个BLWN输入数据集和相应的结构位移和速度响应，被认为是训练或验证的“已知”数据集（分割比为0.8:0.2）。此外，50个额外的辅助样本（例如仅BLWN输入记录）用于指导具有物理约束的模型训练。
- 测试数据：其余数据集被视为“未知”数据集，以测试训练元模型的预测性能。

## [环境要求](#目录)

- 硬件（Ascend）
    - 使用 Ascend处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 欲了解更多信息，请查看以下资源:
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速开始](#目录)

通过官网安装好MindSpore和上面需要的[数据集](#数据集)后，就可以开始训练和测试如下:

- 在 Ascend上运行

默认:

```bash
python train.py
```

完整命令:

```bash
python train.py \
    --print_interval 1 \
    --ckpt_interval 1 \
    --save_fig true \
    --save_ckpt true \
    --load_ckpt false \
    --save_ckpt_path ./checkpoints \
    --load_ckpt_path ./checkpoints/your_file.ckpt \
    --figures_path ./figures \
    --load_data_path ./data/data_boucwen.mat \
    --log_path ./logs \
    --epochs 8000 \
    --lr 1e-4 \
    --mode train
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── PHYLSTM
│   ├── checkpoints                                    # checkpoint文件
│   ├── data                                           # 数据文件
│   │   └── data_boucwen.mat                           # SDOF Bouc-Wen数据集
|   ├── docs
│   │   └──Physics-informed double-LSTM Network.png    # PHYLSTM2网络结构图
│   ├── figures                                        # 结果图片
│   ├── logs                                           # 训练记录文件
│   ├── src                                            # 源代码
│   │   ├── network.py                                 # 网络架构
│   │   ├── plot.py                                    # 绘制结果
│   │   └── process.py                                 # 数据处理
│   ├── config.json                                    # 超参数配置
│   ├── README.md                                      # 英文模型说明
│   ├── README_CN.md                                   # 中文模型说明
│   ├── test.py                                        # python测试脚本
│   └── train.py                                       # python训练脚本
```

### [脚本参数](#目录)

`train.py`中的重要参数如下:

| 参数               | 描述                     | 默认值                          |
|------------------|------------------------|------------------------------|
| print_interval   | 时间与loss打印间隔          | 1                          |
| ckpt_interval    | checkpoint保存周期         | 1                          |
| save_fig         | 是否保存和绘制图片              | true                         |
| save_ckpt        | 是否保存checkpoint         | true                         |
| load_ckpt        | 是否加载checkpoint         | false                        |
| save_ckpt_path   | checkpoint保存路径         | ./checkpoints                |
| load_ckpt_path   | checkpoint加载路径         | ./checkpoints/your_file.ckpt |
| figures_path     | 图片保存路径                 | ./figures                    |
| load_data_path   | 加载验证数据的路径              | ./data                       |
| log_path         | 日志保存路径                 | ./logs                       |
| epochs           | 训练周期                   | 8000                        |
| lr               | 学习率                    | 1e-4                         |
| mode        | 运行模式          | train                      |

### [训练流程](#目录)

- 在 Ascend 上运行

   ```bash
   python train.py
   ```

  训练期间的损失值将打印在控制台中。

  ```bash
  [Adam]Epoch:0,Train_Loss:2.2092102,Eval_Loss:2.170113,bestLoss:2.209210157394409,bestEpoch:0
  [Adam]Epoch:1,Train_Loss:2.1713018,Eval_Loss:2.127627,bestLoss:2.17130184173584,bestEpoch:1
  [Adam]Epoch:2,Train_Loss:2.1313734,Eval_Loss:2.1100757,bestLoss:2.131373405456543,bestEpoch:2
  [Adam]Epoch:3,Train_Loss:2.102894,Eval_Loss:2.065416,bestLoss:2.1028940677642822,bestEpoch:3
  [Adam]Epoch:4,Train_Loss:2.0730639,Eval_Loss:2.0375416,bestLoss:2.073063850402832,bestEpoch:4
  [Adam]Epoch:5,Train_Loss:2.043288,Eval_Loss:2.019644,bestLoss:2.043287992477417,bestEpoch:5
  [Adam]Epoch:6,Train_Loss:2.013958,Eval_Loss:2.0036902,bestLoss:2.013957977294922,bestEpoch:6
  [Adam]Epoch:7,Train_Loss:1.9865956,Eval_Loss:1.9802157,bestLoss:1.986595630645752,bestEpoch:7
  [Adam]Epoch:8,Train_Loss:1.9615813,Eval_Loss:1.945712,bestLoss:1.9615813493728638,bestEpoch:8
  [Adam]Epoch:9,Train_Loss:1.9371263,Eval_Loss:1.9073497,bestLoss:1.9371262788772583,bestEpoch:9
  ...
  ```

- 训练结束后，您仍然可以通过保存在`log_path`下面的日志文件回顾训练中损失函数的过程，默认为`./logs`目录中。

- 模型checkpoint将保存在 `save_ckpt_path`中，默认为`./checkpoints` 目录中。

### [推理流程](#目录)

在运行下面的命令之前，请检查使用的`config.json` 中的checkpoint加载路径`load_ckpt_path`
进行推理。

- 在 Ascend 上运行

   ```bash
   python eval.py
   ```

- 您可以通过终端查看过程与结果。
- 结果图片存放于`figures_path`中，默认位于`./figures`。