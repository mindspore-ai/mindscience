<!-- markdownlint-disable first-line-h1 -->

English | [简体中文](README_CN.md)

# MindElec

- [MindElec](#mindelec)
    - [Introduction to MindElec](#introduction-to-mindelec)
    - [Latest News](#latest-news)
    - [Application Cases](#application-cases)
    - [Related Papers](#related-papers)
    - [Core Contributors](#core-contributors)
    - [Contribution Guidelines](#contribution-guidelines)
    - [License](#license)

## Introduction to MindElec

Electromagnetic simulation refers to simulating the propagation characteristics of electromagnetic waves in objects or space through computation. It is widely applied in scenarios such as mobile phone tolerance, antenna optimization, and chip design. Traditional numerical methods, such as finite difference and finite element methods, require mesh partitioning and iterative computation, resulting in complex simulation processes and long computation times, which cannot meet product design requirements. AI methods have universal approximation and efficient inference capabilities, which can effectively improve simulation efficiency.

MindElec is an AI electromagnetic simulation toolkit developed based on MindSpore, consisting of data construction and transformation, simulation computation, and result visualization. It supports end-to-end AI electromagnetic simulation. It has already achieved phased results in the Huawei terminal mobile phone tolerance scenario. Compared with commercial simulation software, the S-parameter error of AI electromagnetic simulation is about 2%, and the end-to-end simulation speed is improved by more than 10 times.

## Latest News

## Application Cases

|                   Case                    |    Description   |    Model Architecture   |
| :---------------------------------------: | :-------: | :---------: |
| [AI Solver for Time-Domain Maxwell Equations with Point Source][time_domain_maxwell-URL] | Solving 2D time-domain Maxwell equations based on PINNs method through Gaussian distribution function smoothing, multi-channel residual networks combined with sin activation function, and adaptive weighted multi-task learning strategy | Multi-channel residual networks combined with sin activation function |
| [Incremental Training for Solving Maxwell Equations][incremental_learning-URL] | Using physics-informed auto-decoder to map high-dimensional variable parameter space to low-dimensional manifold, solving equations with different parameters through pre-trained model fine-tuning | Physics-informed auto-decoder |
| [AI Electromagnetic Simulation Based on Parameterization Scheme][parameterization-URL] | Achieving direct mapping from antenna parameters (such as width, angle) to scattering parameters (S parameters) | Direct mapping network from parameters to simulation results |
| [AI Electromagnetic Simulation Based on Point Cloud Scheme][point_cloud-URL] | Converting mobile phone structure files to point cloud tensor data, using convolutional neural networks to extract structural features and mapping to S parameters | Convolutional neural network for feature extraction + fully connected layer mapping |
| [Patch Antenna S Parameter Simulation Based on Differentiable FDTD][AD_FDTD_forward-URL] | Rewriting FDTD update process with MindSpore's differentiable operators to achieve end-to-end differentiable FDTD for S parameter simulation | Recurrent convolutional network (RCNN) |
| [End-to-End Differentiable FDTD for Electromagnetic Inverse Scattering Problems][AD_FDTD_inverse-URL] | Solving 2D TM mode electromagnetic inverse scattering problems based on end-to-end differentiable FDTD, achieving high-precision permittivity inversion | End-to-end differentiable FDTD network |

## Related Papers

If you are interested in solving time-domain Maxwell equations, please read our related [paper](https://arxiv.org/abs/2111.01394): Xiang Huang, Hongsheng Liu, Beiji Shi, Zidong Wang, Kang Yang, Yang Li, Bingya Weng, Min Wang, Haotian Chu, Jing Zhou, Fan Yu, Bei Hua, Lei Chen, Bin Dong, Solving Partial Differential Equations with Point Source Based on Physics-Informed Neural Networks, preprint 2021

If you are interested in meta-learning auto-decoder for solving parametric partial differential equations, please read our related [paper](https://arxiv.org/abs/2111.08823): Xiang Huang, Zhanhong Ye, Hongsheng Liu, Beiji Shi, Zidong Wang, Kang Yang, Yang Li, Bingya Weng, Min Wang, Haotian Chu, Jing Zhou, Fan Yu, Bei Hua, Lei Chen, Bin Dong, Meta-Auto-Decoder for Solving Parametric Partial Differential Equations, preprint 2021

## Core Contributors

Thanks to the following developers for their contributions to MindElec:

## Contribution Guidelines

Welcome to contribute your code to MindElec by referring to the [Contribution Guidelines](../CONTRIBUTION.md)!

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

[time_domain_maxwell-URL]: https://www.mindspore.cn/mindelec/docs/en/master/time_domain_maxwell.html
[incremental_learning-URL]: https://www.mindspore.cn/mindelec/docs/en/master/incremental_learning.html
[parameterization-URL]: https://www.mindspore.cn/mindelec/docs/en/master/parameterization.html
[point_cloud-URL]: https://www.mindspore.cn/mindelec/docs/en/master/point_cloud.html
[AD_FDTD_forward-URL]: https://www.mindspore.cn/mindelec/docs/en/master/AD_FDTD_forward.html
[AD_FDTD_inverse-URL]: https://www.mindspore.cn/mindelec/docs/en/master/AD_FDTD_inverse.html