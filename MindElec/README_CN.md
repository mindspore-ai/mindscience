<!-- markdownlint-disable first-line-h1 -->

简体中文 | [English](README.md)

# MindElec

- [MindElec](#mindelec)
    - [MindElec 介绍](#mindelec-介绍)
    - [最新消息](#最新消息)
    - [应用案例](#应用案例)
    - [相关论文](#相关论文)
    - [核心贡献者](#核心贡献者)
    - [贡献指南](#贡献指南)
    - [许可证](#许可证)

## MindElec 介绍

电磁仿真是指通过计算的方式模拟电磁波在物体或空间中的传播特性，其在手机容差、天线优化和芯片设计等场景中应用广泛。传统数值方法如有限差分、有限元等需网格剖分、迭代计算，仿真流程复杂、计算时间长，无法满足产品的设计需求。AI方法具有万能逼近和高效推理能力，可有效提升仿真效率。

MindElec 是基于 MindSpore 开发的AI电磁仿真工具包，由数据构建及转换、仿真计算、以及结果可视化组成。可以支持端到端的 AI 电磁仿真。目前已在华为终端手机容差场景中取得阶段性成果，相比商业仿真软件，AI 电磁仿真的 S 参数误差在2%左右，端到端仿真速度提升10+倍。

## 最新消息

## 应用案例

|                   案例                    |    简介   |    模型架构   |
| :---------------------------------------: | :-------: | :---------: |
| [点源时域麦克斯韦方程AI求解][time_domain_maxwell-URL] | 基于PINNs方法求解二维时域MaxWell方程，通过高斯分布函数平滑、多通道残差网络结合sin激活函数以及自适应加权的多任务学习策略实现 |  多通道残差网络结合sin激活函数 |
| [增量训练求解麦克斯韦方程组][incremental_learning-URL] | 采用基于物理信息的自解码器，将高维可变参数空间映射到低维流形，通过预训练模型微调解决不同参数的方程组 | 基于物理信息的自解码器 |
| [基于参数化方案的AI电磁仿真][parameterization-URL] | 实现天线参数（如宽度、角度）到散射参数（S参数）的直接映射 | 参数到仿真结果的直接映射网络 |
| [基于点云方案的AI电磁仿真][point_cloud-URL] | 将手机结构文件转化为点云张量数据，使用卷积神经网络提取结构特征并映射到S参数 | 卷积神经网络提取结构特征+全连接层映射 |
| [基于可微分FDTD的贴片天线S参数仿真][AD_FDTD_forward-URL] | 利用MindSpore的可微分算子重写FDTD更新流程，实现端到端可微分FDTD进行S参数仿真 | 循环卷积网络（RCNN） |
| [端到端可微分FDTD求解电磁逆散射问题][AD_FDTD_inverse-URL] | 基于端到端可微FDTD求解二维TM模式的电磁逆散射问题，实现高精度的介电常数反演 | 端到端可微分FDTD网络 |

## 相关论文

如果你对求解时域麦克斯韦方程感兴趣，请阅读我们的相关[论文](https://arxiv.org/abs/2111.01394): Xiang Huang, Hongsheng Liu, Beiji Shi, Zidong Wang, Kang Yang, Yang Li, Bingya Weng, Min Wang, Haotian Chu, Jing Zhou, Fan Yu, Bei Hua, Lei Chen, Bin Dong, Solving Partial Differential Equations with Point Source Based on Physics-Informed Neural Networks, preprint 2021

如果你对元学习自解码器求解参数化偏微分方程感兴趣，请阅读我们的相关[论文](https://arxiv.org/abs/2111.08823): Xiang Huang, Zhanhong Ye, Hongsheng Liu, Beiji Shi, Zidong Wang, Kang Yang, Yang Li, Bingya Weng, Min Wang, Haotian Chu, Jing Zhou, Fan Yu, Bei Hua, Lei Chen, Bin Dong, Meta-Auto-Decoder for Solving Parametric Partial Differential Equations, preprint 2021

## 核心贡献者

感谢以下开发者对 MindElec 的贡献：

## 贡献指南

欢迎参考[贡献指南](../CONTRIBUTION.md)为 MindElec 贡献您的代码！

## 许可证

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

[time_domain_maxwell-URL]: https://www.mindspore.cn/mindelec/docs/zh-CN/master/time_domain_maxwell.html
[incremental_learning-URL]: https://www.mindspore.cn/mindelec/docs/zh-CN/master/incremental_learning.html
[parameterization-URL]: https://www.mindspore.cn/mindelec/docs/zh-CN/master/parameterization.html
[point_cloud-URL]: https://www.mindspore.cn/mindelec/docs/zh-CN/master/point_cloud.html
[AD_FDTD_forward-URL]: https://www.mindspore.cn/mindelec/docs/zh-CN/master/AD_FDTD_forward.html
[AD_FDTD_inverse-URL]: https://www.mindspore.cn/mindelec/docs/zh-CN/master/AD_FDTD_inverse.html
