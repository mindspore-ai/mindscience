# 利用PINNs求解泊松方程

## 问题描述

本案例演示如何利用PINNs在不同几何体下求解二维和三维泊松方程。二维泊松方程定义为

$$
\Delta u = -\sin(4\pi x)\sin(4\pi y),
$$

而三维方程定义为

$$
\Delta u = -\sin(4\pi x)\sin(4\pi y)\sin(4\pi z),
$$

很容易验证，以下函数分别满足二维和三维泊松方程

$$
u = \frac{1}{32\pi^2} \sin(4\pi x)\sin(4\pi y), \\
u = \frac{1}{48\pi^2} \sin(4\pi x)\sin(4\pi y)\sin(4\pi z).
$$

如果在几何体边界按以上函数取狄利克雷边界条件，那么这些函数就是我们想要得到的解。因而，我们可以利用以上函数来验证结果。
对于二维问题，本例演示在矩形，圆形，三角形和五边形区域求解方程，而对于三维问题，我们将在四面体，圆柱和圆锥区域内求解方程。

## 技术路径

MindFlow求解该问题的具体流程如下：

1. 创建训练数据集。
2. 构建模型。
3. 优化器。
4. 约束。
5. 模型训练。
6. 模型评估。

## 创建数据集

本案例在求解域及边值条件进行随机采样，生成训练数据集与测试数据集。具体方法见``src/dataset.py``。

## 构建模型

本案例采用带3个隐藏层的多层感知器，并带有以下特点:

- 采用激活函数：$f(x) = x \exp(-x^2/(2e)) $

- 最后一层线性层使用weight normalization。

- 所有权重都采用``mindspore``的``HeUniform``初始化。

具体定义见``src/model.py``。

## 优化器

本案例采用Adam优化器，并配合[Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)提出的动态学习率进行训练。
在求解区域内和边界上均采用L2损失，并利用``mindflow``的``MTLWeightedLossCell``多目标损失函数将两个损失结合起来。具体定义见``train.py``，动态学习率定义参见``src/lr_scheduler.py``。

## 约束

在利用``mindflow``求解PDE时，我们需要写一个``mindflow.pde.Problem``的子类来定义控制方程和边界条件。读者可在``src/possion.py``内找到具体实现。

## 模型训练

运行

 ``python train.py GEOMETRY --ckpt_dir=CKPT_DIR --n_epoch=XXX``

其中``GEOMETRY``为几何体名称，可选择rectangle, disk, triangle, pentagon, tetrahedon, cylinder和cone。``CKPT_DIR``为checkpoints保存路径。

## 模型评估

运行

 ``python eval.py GEOMETRY CHECKPOINT``

其中``GEOMETRY``为几何体名称，可选择rectangle, disk, triangle, pentagon, tetrahedon, cylinder和cone。 几何体需要与训练时使用的相匹配。``CHECKPOINT``为checkpoints的文件名。

## 模型精度

如果训练600个epochs，模型可以到达的精度如下表所示：
|        | Domain        | Boundary      |
| ------ | ------------- | ------------- |
| 矩形   | 0.04% (0.04%) | NA            |
| 圆形   | 0.05% (0.05%) | 0.22% (0.21%) |
| 三角形 | 0.10% (0.10%) | 0.38% (0.38%) |
| 五边形 | 0.08% (0.08%) | 0.29% (0.28%) |

|        | Domain        | Boundary      |
| ------ | ------------- | ------------- |
| 四面体 | 0.51% (0.51%) | 1.26% (1.25%) |
| 圆柱   | 1.03% (1.04%) | 5.00% (5.05%) |
| 圆锥   | 0.68% (0.71%) | 2.28% (2.31%) |

表格内均为相对L2误差，括号内的是在测试集上的误差。矩形区域的解在边界上均为0，因此没有L2相对误差。
