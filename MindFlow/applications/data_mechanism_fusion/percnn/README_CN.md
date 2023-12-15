[ENGLISH](README.md) | 简体中文

# 物理嵌入递归卷积神经网络（Physics-encoded Recurrent Convolutional Neural Network，PeRCNN）

## 概述

PDE方程在对物理系统的建模中占据着中心地位，但在流行病学、气象科学、流体力学和生物学等等领域中，很多的底层PDE仍未被完全发掘出来。而对于那些已知的PDE方程，比如Naiver-Stokes方程，对这些方程进行精确数值计算需要巨大的算力，阻碍了数值仿真了在大规模时空系统上的应用。目前，机器学习的进步提供了一种PDE求解和反演的新思路。

![PeRCNN](images/percnn.jpg)

近日，华为与中国人民大学孙浩教授团队合作，基于昇腾AI基础软硬件平台与昇思
MindSpore AI框架提出了一种[物理编码递归卷积神经网络（Physics-encoded Recurrent Convolutional Neural Network， PeRCNN）](https://www.nature.com/articles/s42256-023-00685-7)。相较于物理信息神经网络、ConvLSTM、PDE-NET等方法，模型泛化性和抗噪性明显提升，长期推理精度提升了
10倍以上，在航空航天、船舶制造、气象预报等领域拥有广阔的应用前景，目前该成果已在 nature machine intelligence 上发表。

PeRCNN神经网络强制嵌入物理结构，基于结合部分物理先验设计的π-卷积模块，通过特征图之间的元素乘积实现非线性逼近。该物理嵌入机制保证模型根据我们的先验知识严格服从给定的物理方程。所提出的方法可以应用于有关PDE系统的各种问题，包括数据驱动建模和PDE的发现，并可以保证准确性和泛用性。

## 案例

- [2d burgers方程](./burgers_2d/)

- [3d 反应扩散方程](./gsrd_3d/)