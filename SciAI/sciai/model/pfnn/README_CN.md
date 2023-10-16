[ENGLISH](README.md) | 简体中文

## 目录

- [PFNN 描述](#PFNN-描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练流程](#训练流程)
    - [推理流程](#推理流程)

## [PFNN 描述](#目录)

PFNN (Penalty-free neural network)方法是一种基于神经网络的微分方程求解方法，适用于求解复杂区域上的二阶微分方程。该方法克服了已有类似方法在处理问题光滑性约束和边界约束上的缺陷，具有更高的精度，效率和稳定性。

[论文](https://www.sciencedirect.com/science/article/pii/S0021999120308597)：H. Sheng, C. Yang, PFNN: A penalty-free neural network method for solving a class of second-order boundary-value problems on complex geometries, Journal of Computational Physics 428 (2021) 110085.

## [模型架构](#目录)

PFNN采用神经网络逼近微分方程的解。不同于大多数只采用单个网络构造解空间的神经网络方法，PFNN采用两个网络分别逼近本质边界和区域其它部分上的真解。为消除两个网络之间的影响，一个由样条函数所构造的length factor函数被引入以分隔两个网络。为进一步降低问题对于解的光滑性需求，PFNN利用Ritz变分原理将问题转化为弱形式，消除损失函数中的高阶微分算子，从而降低最小化损失函数的困难，有利于提高方法的精度。

## [数据集](#目录)

PFNN根据方程信息和计算区域信息生成训练集和测试集。

- 训练集：分为内部集和边界集，分别在计算区域内部和边界上采样得到。
    - 内部集：在计算区域内部采样3600个点，并计算控制方程右端项在这些点上的值作为标签。
    - 边界集：在Dirichlet边界和Neumann边界上分别采样60和180个点，并计算边界方程右端项在这些点上的值作为标签。

- 测试集：在整个计算区域上采样10201个点，并计算真解在这些点上的值作为标签。

    注：该数据集在各向异性的扩散方程场景中使用。数据将在pfnn/Data/Data.py中处理

## [环境要求](#目录)

- 硬件（Ascend/GPU/CPU）
- 框架
    - [Mindspore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [Mindspore教程](#https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [Mindspore Python API](#https://www.mindspore.cn/docs/zh-CN/master/index.html)

## [快速开始](#目录)

通过官网安装好MindSpore后，就可以开始训练和验证如下:

- 在 Ascend 或 GPU 上运行

默认:

```bash
python train.py
```

## [脚本说明](#目录)

### [脚本和示例代码](#目录)

文件结构如下:

```text
├── data
│   ├── gendata.py             # 根据方程生成数据
│   └── dataset.py             # 生成数据集
├── src
│   ├── callback.py            # 回调模块
│   ├── process.py             # 训练准备
│   └── pfnnmodel.py           # 网络模型
├── README_CN.md               # 模型中文说明
├── README.md                  # 模型英文说明
├── config.yaml                # 超参数配置
├── train.py                   # python训练脚本
└── eval.py                    # python验证脚本
```

### [脚本参数](#目录)

训练和评估的超参数设置可以在 `config.yaml` 文件中进行配置

| 参数             | 描述                         | 默认值                                                                                |
|----------------|----------------------------|------------------------------------------------------------------------------------|
| problem_type   | 问题类型                       | 1                                                                                  |
| bound          | 边界                         | [-1.0, 1.0, -1.0, 1.0]                                                             |
| inset_nx       | InnerSet数量                 | [60, 60]                                                                           |
| bdset_nx       | BoundarySet数量              | [60, 60]                                                                           |
| teset_nx       | TestSet数量                  | [101, 101]                                                                         |
| g_epochs       | g_net时期（迭代次数）              | 6000                                                                               |
| f_epochs       | f_net时期（迭代次数）              | 6000                                                                               |
| g_lr           | g_net学习率                   | 0.01                                                                               |
| f_lr           | f_net学习率                   | 0.01                                                                               |
| tests_num      | 测试次数                       | 5                                                                                  |
| log_path       | 日志保存路径                     | ./logs                                                                             |
| load_ckpt_path | checkpoint路径               | [./checkpoints/optimal_state_g_pfnn.ckpt, ./checkpoints/optimal_state_f_pfnn.ckpt] |
| force_download | 是否强制下载数据集与checkpoint       | false                                                                              |
| amp_level      | MindSpore自动混合精度等级          | O0                                                                                 |
| mode           | MindSpore静态图模式（0）或动态图模式（1） | 0                                                                                  |

### [训练流程](#目录)

  ```bash
  python train.py --problem_type [PROBLEM] --g_epochs [G_EPOCHS] --f_epochs [F_EPOCHS] --g_lr [G_LR] --f_lr [F_LR]
  ```

### [推理流程](#目录)

 在运行下面的命令之前，请检查使用的`config.yaml` 中的checkpoint加载路径`load_ckpt_path`进行推理。

- 在 GPU/Ascend 上运行

   ```bash
   python eval.py
   ```
