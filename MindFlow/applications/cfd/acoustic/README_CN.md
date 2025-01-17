# 2D 声波方程 CBS 求解

## 概述

声波方程求解是医疗超声、地质勘探等领域中的核心技术，大规模声波方程求解面临算力和存储的挑战。声波方程求解器一般采用频域求解算法和时域求解算法，时域求解算法的代表是时域有限差分法 (TDFD)，频域求解算法包括频域有限差分法 (FDFD)、有限元法 (FEM) 和 CBS (Convergent Born series) 迭代法。CBS 方法由于不引入频散误差，且求解的内存需求低，因此受到工程和学术界的广泛关注。尤其是 [Osnabrugge et al. (2016)](https://linkinghub.elsevier.com/retrieve/pii/S0021999116302595) 解决了该方法的收敛性问题，使得 CBS 方法的应用具有更广阔的前景。基于 CBS 的计算结构所提出的 AI 模型也是物理与 AI 双驱动范式的典型代表，包括 [Stanziola et al. (2022)](http://arxiv.org/abs/2212.04948)，[Zeng et al. (2023)](http://arxiv.org/abs/2312.15575) 等。

本案例将演示如何调用 MindFlow 提供的 CBS API 实现二维声波方程的求解。

## 理论背景

### 问题描述

声波方程求解中，波速场和震源信息是输入参数，求解输出的是时空分布的波场。

二维声波方程的表达式如下

| 时域表达式                                            | 频域表达式                                        |
| ----------------------------------------------------- | ------------------------------------------------- |
| $\frac{\partial^2u}{\partial t^2} - c^2 \Delta u = f$ | $-\omega^2 \hat{u} - c^2 \Delta\hat{u} = \hat{f}$ |

其中

- $u(\bold{x},t) \;\; [L]$ 变形位移 (压强除以密度)，标量
- $c(\bold{x}) \;\; [L/T]$ 波速，标量
- $f(\bold{x},t) \;\; [L/T^2]$ 震源激励 (体积分布力)，标量

实际求解中，为了降低参数维度，一般先将参数无量纲化，然后针对无量纲方程和参数进行求解，最后恢复解的量纲。选取 $\omega$、$\hat{f}$ 和 $d$（网格间距，本案例要求网格在各方向间距相等）对频域方程做无量纲化，可得频域无量纲方程：

$$
u^* + c^{*2} \tilde{\Delta} + f^* = 0
$$

其中

- $u^* = \hat{u} \omega^2 / \hat{f}$ 为无量纲变形位移
- $c^* = c / (\omega d)$ 为无量纲波速
- $\tilde{\Delta}$ 为归一化 Laplace 算子，即网格间距均为 1 时的 Laplace 算子
- $f^*$ 为标记震源位置的 mask，即在震源作用点为 1，其余位置为 0

### CBS 方法介绍

此处对 CBS 方法的理论推导作简单介绍，读者可参考 [Osnabrugge et al. (2016)](https://linkinghub.elsevier.com/retrieve/pii/S0021999116302595) 进一步了解。

**原始 Born Series**

首先将频域声波方程表达为以下等价形式
$$
k^2 \hat{u} + \Delta \hat{u} +S = 0
$$
其中 $k=\omega/c$，$S=\hat{f}/c^2$。将非均匀波数场 $k$ 拆分为均匀背景势 $k_0$ 和散射势 $V$：$k^2 = V + k_0^2 + i\epsilon$，其中  $\epsilon$ 为保持迭代稳定的小量，方程的最终解与 $k_0, \epsilon$ 的具体取值无关。得到单次迭代求解的方程
$$
(k_0^2 + i\epsilon) \hat{u} + \Delta \hat{u} = -V \hat{u} - S
$$
将右端项视为已知量，该方程的解为
$$
\hat{u} = G (V \hat{u} + S)
\qquad
G = \mathcal{F}^{-1} \frac1{|\bold{p}|^2 - k_0^2 - i\epsilon} \mathcal{F}
$$
将每轮迭代的求解结果代回右端项，进行下一轮迭代，得迭代表达式
$$
\hat{u}_{k+1} = GV\hat{u}_k + GS = (1 + GV + GVGV + \cdots)GS
$$
**收敛 Born Series**

为了保证收敛性，需做一定预处理以及合理选取 $\epsilon$ 的值。定义预处理子 $\gamma = \frac{i}{\epsilon} V$，并取 $\epsilon \ge \max{|k^2 - k_0^2|}$，将迭代的等式两端同乘 $\gamma$ 并整理，可得
$$
\hat{u} = (\gamma GV - \gamma + 1) \hat{u} + \gamma GS
$$
记 $\gamma GV - \gamma + 1 = M$，则迭代式变为
$$
\hat{u}_{k+1} = M \hat{u}_k + \gamma GS = (1 + M + M^2 + \cdots) \gamma GS
$$
矩阵形式
$$
\begin{bmatrix} \hat{u}_k \\ S \end{bmatrix} =
\begin{bmatrix} M & \gamma G \\ 0 & 1 \end{bmatrix}^k
\begin{bmatrix} 0 \\ S \end{bmatrix}
$$
实际程序植入时，为了减少 Fourier 变换的次数，采用以下等价形式的迭代式
$$
\hat{u}_{k+1} = \hat{u}_k + \gamma [G(V\hat{u}_k + S) - \hat{u}_k]
$$

## 案例设计

具体包含以下步骤

- 输入参数无量纲化；
- 频域无量纲化 2D 声波方程 CBS 求解；
- 求解结果恢复量纲化；
- 求解结果时频转换。

其中核心求解的过程针对不同震源位置和不同频点同时并行求解，由于频点数可能较多，因此沿频率方向分为 `n_batches` 个批次依次求解。

案例所需的输入以文件的形式放置于 `dataset/` 中，文件名通过 `config.yaml` 传入。输出的结果为频域无量纲方程的解 `u_star.npy`、转换到时域的有量纲最终解 `u_time.npy`、针对时域解制作的可视化动图 `wave.gif`。

## 快速开始

为了方便用户直接验证，本案例在本[链接](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/cfd/acoustic)中提供了预置的输入数据，请下载所需要的数据集，并保存在 `./dataset` 目录下。数据集包括速度场 `velocity.npy`、震源位置列表 `srclocs.csv`、震源波形 `srcwaves.csv`。用户可仿照输入文件格式自行修改输入参数。

### 运行方式一：`solve_acoustic.py` 脚本

```shell
python solve_acoustic.py --config_file_path ./configs.yaml --device_id 0 --mode GRAPH
```

其中，

`--config_file_path`表示配置文件的路径，默认值'./config.yaml'；

`--device_id`表示使用的计算卡编号，可按照实际情况填写，默认从所有计算卡中自动选取最空闲的一张；

`--mode`表示运行的模式，'GRAPH'表示静态图模式, 'PYNATIVE'表示动态图模式。

### 运行方式二：运行 Jupyter Notebook

您可以使用[中文版](./acoustic_CN.ipynb)和[英文版](./acoustic.ipynb)Jupyter Notebook 逐行运行训练和验证代码。

## 结果展示

针对同一个速度模型，不同震源位置激发的波场随时间演化过程如下图所示。

![wave.gif](images/wave.gif)

方程残差的迭代收敛过程如下图所示，每根线代表一个频点。不同频点达到收敛阈值所需的迭代次数不同，同一批次的迭代次数取决于收敛最慢的频点。

![errors.png](images/errors.png)

## 性能

| 参数               | Ascend               |
|:----------------------:|:--------------------------:|
| 硬件资源                | 昇腾 NPU            |
| MindSpore版本           | >=2.3.0                 |
| 数据集                  | [Marmousi 速度模型](https://en.wikipedia.org/wiki/Marmousi_model)切片，包含在案例 `dataset/` 路径中 |
| 参数量                  | 无可学习参数   |
| 求解参数              | batch_size=300, tol=1e-3, max_iter=10000 |
| 收敛所需迭代数  | batch 0: 1320, batch 1: 560, batch 2: 620, batch 3: 680|
| 求解速度(ms/iteration) | 500                 |

## 贡献者

gitee id: [WhFanatic](https://gitee.com/WhFanatic)

email: hainingwang1995@gmail.com