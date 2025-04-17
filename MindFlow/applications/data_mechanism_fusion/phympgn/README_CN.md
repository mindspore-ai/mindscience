# 用于时空PDE系统的物理编码消息传递图神经网络 (PhyMPGN: Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems)

偏微分方程（PDEs）控制的复杂动力系统广泛存在于各个学科当中。近年来，数据驱动的神经网络模型在预测时空动态上取得了极好的效果。

物理编码的消息传递图网络（PhyMPGN），可以使用少量训练数据在不规则计算域上建模时空PDE系统。具体来说，

- 提出了一个使用消息传递机制的物理编码图学习模型，使用二阶龙格库塔（Runge-Kutta）数值方案进行时间步进
- 考虑到物理现象中普遍存在扩散过程，设计了一个可学习的Laplace Block，编码了离散拉普拉斯-贝尔特拉米算子（Laplace-Beltrami Operator）
- 提出了一个新颖的填充策略在模型中编码不同类型的边界条件

论文链接: [https://arxiv.org/abs/2410.01337](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F2410.01337)。

该论文被接收为 ICLR 2025 Spotlight，详见 https://openreview.net/forum?id=fU8H4lzkIm&noteId=wS5SaVKjWt。

## 问题描述

考虑由如下形式控制的时空PDE系统：

$$
\begin{equation}
\dot {\boldsymbol {u}}(\boldsymbol x, t) = \boldsymbol F (t, \boldsymbol x, \boldsymbol u, \nabla \boldsymbol u, \Delta \boldsymbol u, \dots)
\end{equation}
$$

其中$\boldsymbol u(\boldsymbol x, y) \in \mathbb{R}^m$是具有$m$个分量的状态变量向量，例如速度、温度或者压力等，它的定义在时空域$\{ \boldsymbol x, t \} \in \Omega \times [0, \mathcal{T}]$上；$\dot{\boldsymbol u}$代表$\boldsymbol u$对时间的导数，$\boldsymbol F$是依赖于当前状态$\boldsymbol u$和其空间导数的非线性算子。

假设在空间域$\Omega$上有着非均匀且稀疏的观测结点$\{ \boldsymbol x_0, \dots, \boldsymbol x_{N-1} \}$（即，非结构化网格），在时刻$t_0, \dots, t_{T-1}$，这些结点上的观测为$\{ \boldsymbol U(t_0), \dots, \boldsymbol U(t_{T-1}) \}$，其中的$\boldsymbol U(t_i) = \{ \boldsymbol u(\boldsymbol x_0, t_i), \dots, \boldsymbol u (\boldsymbol x_{N-1}, t_i) \}$代表某些物理量。考虑到很多物理现象包含扩散过程，我们假设PDE中的扩散项是已知的先验知识。我们的目标是使用少量训练数据学习一个图神经网络模型，在稀疏非结构网格上预测不同的时空动态系统，处理不同的边界条件，为任意的初始条件产生后续动态轨迹。

## 模型

<img src="images/phympgn.png" alt="Markdown Logo" width="800" />

对于式（1），可以使用二阶龙格库塔（Runge-Kutta, RK2）方案进行离散：

$$
\begin{equation}
\boldsymbol u^{k+1} = \boldsymbol u^k + \frac{1}{2}(\boldsymbol g_1 + \boldsymbol g_2); \quad \boldsymbol g_1 = \boldsymbol F(t^k, \boldsymbol x, \boldsymbol u^k, \dots); \quad \boldsymbol g_2 = \boldsymbol F(t^{k+1}, \boldsymbol x, \boldsymbol u^k + \delta t \boldsymbol g_1, \dots)
\end{equation}
$$

其中$\boldsymbol u^k$为$t^k$时刻的状态变量，$\delta t$为时刻$t^k$和$t^{k+1}$之间的时间间隔。根据式（2），我们构建一个GNN来学习非线性算子$\boldsymbol F$.

如图所示，我们使用NN block来学习非线性算子$\boldsymbol F$。NN block又可以分为两部分：采用编码器-处理器-解码器架构的GNN block和可学习的Laplace block。因为物理现象中扩散过程的普遍存在性，我们设计了可学习的Laplace block，编码离散拉普拉斯贝尔特拉米算子（Laplace-Beltrami operator），来学习由PDE中扩散项导致的增量；而GNN block来学习PDE中其他项导致的增量。

## 相关依赖库

- python 3.11
- mindspore 2.5.0
- numpy 1.26

## 数据集

该[数据集](https://download-mindspore.osinfra.cn/mindscience/mindflow/dataset/applications/data_mechanism_fusion/PhyMPGN/)包含圆柱绕流的模拟数据，以HDF5格式存储，包括几何结构、流体属性和流动动力学信息，下载后请保存在 `data/2d_cf`目录下。数据集分为训练集和测试集：

- **训练集**：`train_cf_4x2000x1598x2.h5` 包含4条轨迹。
- **测试集**：`test_cf_9x2000x1598x2.h5` 包含9条轨迹。

### 数据格式

每个HDF5文件包含以下属性和数据集：

- `f.attrs['x_c'], f.attrs['y_c']`：**浮点数**，圆柱中心的坐标。
- `f.attrs['r']`：**浮点数**，圆柱的半径。
- `f.attrs['x_l'], f.attrs['x_r'], f.attrs['y_b'], f.attrs['y_t']`：**浮点数**，计算域的边界。
- `f.attrs['mu']`：**浮点数**，流体粘度。
- `f.attrs['rho']`：**浮点数**，流体密度。
- `f['pos']`：**(n, 2)**，观测节点的位置。
- `f['mesh']`：**(n_tri, 3)**，观测节点的三角网格。
- `g = f['node_type']`：节点类型信息。
    - `g['inlet']`：**(n_inlet,)**，入口边界节点的索引。
    - `g['cylinder']`：**(n_cylinder,)**，圆柱边界节点的索引。
    - `g['outlet']`：**(n_outlet,)**，出口边界节点的索引。
    - `g['inner']`：**(n_inner,)**，域内节点的索引。
- `g = f[i]`：第i条轨迹。
    - `g['U']`：**(t, n, 2)**，速度状态。
    - `g['dt']`：**浮点数**，时间步长之间的间隔。
    - `g['u_m']`：**浮点数**，入口速度。

## 使用方法

`yamls/train.yaml`是项目的配置文件，包括数据集的大小、模型参数和日志、权重保存路径等设置。

**训练**

```python
python main.py --config_file_path yamls/train.yaml --train
```

**测试**

```python
python main.py --config_file_path yamls/train.yaml
```

## 结果展示

$Re=480$

<img src="images/cf.png" alt="Markdown Logo" width="400" />

## 性能

| 参数                           | Ascend                         |
| ------------------------------ | ------------------------------ |
| 硬件资源                       | NPU 显存32G                    |
| 版本                           | Minspore 2.5.0                 |
| 数据集                         | Cylinder flow                  |
| 参数量                         | 950k                           |
| 训练参数                       | batch_size=4,<br />epochs=1600 |
| 训练损失<br />(MSE)            | 3.05e-5                        |
| 验证损失<br />(MSE)            | 5.58e-6                        |
| 推理误差<br />(MSE)            | 4.88e-2                        |
| 训练速度<br />(s / epoch)      | 420 s                          |
| 推理速度<br />(s / trajectory) | 174 s                          |
