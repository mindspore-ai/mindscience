简体中文 | [English](README_EN.md)

# MindEnergy

## MindEnergy介绍

传统电力系统仿真严重依赖基于物理定律的精确数学模型（如微分-代数方程组）。尽管这些模型具有严格的数学理论支撑，但在面对现代电力系统的不确定性和超高维度时，则显得计算昂贵且适应性不足。随着人工智能的发展AI赋能科学计算在电力系统故障分析、潮流计算和暂态稳定评估等领域已经成为不可或缺的研发和设计工具，AI模型具有强大的处理高维和非线性问题的能力，且相比传统数学模型具有更高的计算效率和更好的泛化性，它极大地加速了电力系统仿真等能源领域的技术创新，并且带来更高的效率和安全性。

MindEnergy是基于[昇思MindSpore](https://mindspore.cn)AI框架开发的能源领域套件，该套件集成了电力系统等能源场景的经典AI案例，利用业界主流模型解决电力系统中的潮流计算和暂态分析等问题。MindEnergy提供了主流模型的实现代码及多样化场景下的训练脚本，旨在助力广大科研人员、工程师、高校师生等开发者群体及时掌握人工智能在电力系统等能源领域科学计算中的前沿应用与发展动态，同时也便于对现有模型进行直接部署或二次开发。

## 应用案例

|    案例   |  模型架构   |    硬件    |
|-----------|------------|------------|
|[电力网络动态安全评估](applications/DAE-PINN/README.md)|DAE-PINN|NPU|
|[电力系统故障后预测](applications/DeepONet-Grid-UQ/README.md)|DeepONet-Grid-UQ|NPU|

## 核心贡献者

感谢以下开发者对MindEnergy的贡献：

[@b_rookie](https://gitee.com/b_rookie)，[@congw729](https://gitee.com/congw729)，[@wuzhf9](https://gitee.com/wuzhf9)

## 贡献指南

欢迎参考[贡献指南](https://gitee.com/mindspore/mindscience/blob/br_refactor/CONTRIBUTION.md)为MindEnergy贡献您的代码！

## 许可证

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
