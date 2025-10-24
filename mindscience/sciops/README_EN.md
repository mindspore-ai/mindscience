[简体中文](README.md) | English

# sciops

## Introduction

Traditional AI models typically have regular and dense computational patterns, while AI for science problems are often irregular and sparse. When dealing with irregular scientific computing problems, a large number of computing units are idle, resulting in low utilization of effective computing power. Therefore, AI for science researchers often need to implement efficient and novel domain specific operators.

The sciops module is a high-performance scientific computing operator library in the MindSpore Science framework, which includes some basic scientific computing operators and high-performance fusion operators in scientific computing models. The sciops module provides users with a concise and easy-to-use operator Python API, by calling high-performance operators in the sciops module, users can train models and handle scientific computing tasks more efficiently.

## Operator List

|算子|简介|硬件|
|----|----|----|
|[einsum](einsum)|einsum operator|NPU|
|[evoformer_attention](evoformer_attention)|Ascend-friendly evoformer attention operator|NPU|
|[dft](dft)|discrete fourier transform operator|NPU|
|[fft](fft)|fast fourier transform operator|NPU|

## Contributors

Thanks goes to these wonderful contributors:

[@WhFanatic_admin](https://gitee.com/WhFanatic_admin)，[@machenggui](https://gitee.com/machenggui)，[@alancheng712](https://gitee.com/alancheng712)，[@mulinfro](https://gitee.com/mulinfro)，[@wuzhf9](https://gitee.com/wuzhf9)

## Contribution Guide

Welcome to follow the [contribution guide](https://gitee.com/mindspore/mindscience/blob/br_refactor/CONTRIBUTION.md) to contribute your code for sciops!

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
