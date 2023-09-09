[ENGLISH](README.md) | 简体中文

# 目录

- [GOMO 描述](#Deep-Hpms-描述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [训练过程](#训练过程)
- [模型说明](#模型说明)
    - [评估性能](#评估性能)

## [GOMO 描述](#目录)

海洋广义算子建模（GOMO）是基于[OpenArray v1.0](https://gmd.copernicus.org/articles/12/4729/2019/gmd-12-4729-2019-discussion.html)的三维海洋模型，OpenArray v1.0是用于海洋建模和并行计算解耦的简单算子库（Xiaomeng Huang et al，2019）。GOMO是一种使用有限微分算法求解偏微分方程的数值解模型，使用MindSpore和GPU/Ascend求解这些PDE方程能得到较大的性能提升。
