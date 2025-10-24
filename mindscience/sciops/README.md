简体中文 | [English](README_EN.md)

# sciops

## sciops介绍

传统的AI模型通常具有规整、密集的计算模式，而科学计算问题常常是不规则和稀疏的。在处理这种不规则的科学计算问题时，大量计算单元处于闲置状态，有效算力利用率很低。因此，AI赋能科学计算的研究者经常需要实现高效且全新的、领域特定的算子。

sciops模块是MindSpore Science框架中的高性能科学计算算子库，包含一些基础的科学计算算子和科学计算模型中的高性能融合算子。sciops模块向用户提供简洁易用的算子python API，通过调用sciops模块中的高性能算子，用户能够更加高效地进行模型训练和处理科学计算任务。

## 算子列表

|算子|简介|硬件|
|----|----|----|
|[einsum](einsum)|爱因斯坦求和约定算子|NPU|
|[evoformer_attention](evoformer_attention)|昇腾亲和的生物场景attention算子|NPU|
|[dft](dft)|离散傅里叶变换算子|NPU|
|[fft](fft)|快速傅里叶变换算子|NPU|

## 核心贡献者

感谢以下开发者对sciops模块的贡献：

[@WhFanatic_admin](https://gitee.com/WhFanatic_admin)，[@machenggui](https://gitee.com/machenggui)，[@alancheng712](https://gitee.com/alancheng712)，[@mulinfro](https://gitee.com/mulinfro)，[@wuzhf9](https://gitee.com/wuzhf9)

## 贡献指南

欢迎参考[贡献指南](https://gitee.com/mindspore/mindscience/blob/br_refactor/CONTRIBUTION.md)为sciops模块贡献您的代码！

## 许可证

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
