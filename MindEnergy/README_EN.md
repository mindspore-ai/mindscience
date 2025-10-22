[简体中文](README.md) | English

# MindEnergy

## Introduction

Traditional power system simulation heavily relies on precise mathematical models based on physical laws, such as differential algebraic equation systems. Although these models are very rigorous, they appear computationally expensive and lack adaptability when faced with the uncertainty and ultra-high dimensions of modern power systems. With the development of artificial intelligence, AI for science has become an indispensable research and design tool in fields such as power system fault analysis, power flow calculation, and transient stability assessment. AI models have strong ability to handle high-dimensional and nonlinear problems, and have higher computational efficiency and better generalization compared to traditional mathematical models. It greatly accelerates technological innovation in energy fields such as power system simulation, and brings higher efficiency and safety.

MindEnergy is an energy domain suite developed based on the [MindSpore](https://mindspore.cn) AI framework, which integrates classic AI cases of energy scenarios such as power systems, and uses mainstream models in the industry to solve problems such as power flow calculation and transient analysis in power systems. MindEnergy provides implementation code for mainstream models and training scripts for diverse scenarios, aiming to help developers such as researchers, engineers, university teachers and students timely grasp the cutting-edge applications and development trends of artificial intelligence in scientific computing in energy fields such as power systems. It also facilitates direct deployment or secondary development of existing models.

## Applications

|    案例   |  模型架构   |    硬件    |
|-----------|------------|------------|
|[dynamic security assessment in power grid](applications/DAE-PINN/README.md)|DAE-PINN|NPU|
|[post-fault transient response prediction in power grid](applications/DeepONet-Grid-UQ/README.md)|DeepONet-Grid-UQ|NPU|

## Contributors

Thanks goes to these wonderful contributors:

[@b_rookie](https://gitee.com/b_rookie), [@congw729](https://gitee.com/congw729), [@wuzhf9](https://gitee.com/wuzhf9)

## Contribution Guide

Welcome to follow the [contribution guide](https://gitee.com/mindspore/mindscience/blob/br_refactor/CONTRIBUTION.md) to contribute your code for MindEnergy!

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
