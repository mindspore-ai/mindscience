# 数据集文件

## 预训练数据集

* 方程形式：

$$
\begin{split}
u_t+f_0(u)+s(x)+(f_1(u)-\kappa(x)u_x)_x&=0 , \quad (t,x) \in [0,1] \times [-1,1], \\
u(0,x) &= g(x), \quad x \in [-1,1]
\end{split}
$$

其中 $f_i(u) = c_{i1}u+c_{i2}u^2+c_{i3}u^3$，$i=0,1$。

* 边界条件：周期+非周期。对于非周期边界情形，边界条件种类从 Dirichlet、Neumann 和 Robin 中随机选取，齐次与否也随机选取，且左右两端点的边界条件独立生成。
* 数据集路径：`custom/sinus0`。
* 文件名格式：`custom_v4.23_sinus0_[#bc]_c[#c]_k[#k]_seed[#r].hdf5`
    * 例如：`custom_v4.23_sinus0_robin_cU3_k1e-03_1_seed1.hdf5`
    * `[#bc]`：边界条件类型，`circ` 表示周期边界条件，`robin` 表示非周期边界条件（里面实际上也包括了 Dirichlet、Neumann 这两种退化情形）。
    * `[#c]`：非零系数 $c_{ik}$ 的分布。默认值 `U3`，表示非零系数从分布 $U([-3,3])$ 中采样。
    * `[#k]`：正系数 $\kappa(x)$ 的范围。默认值 `1e-03_1`，表示其最小值为 $10^{-3}$、最大值为 1。
    * `[#r]`：生成数据集文件所用的随机种子，取值为整数。
* 数据生成代码：[../data_generation/custom_sinus.py](../data_generation/custom_sinus.py)。

## 与预训练方程对应的反问题数据集

* 方程形式、边界条件与预训练数据集相同。
    每个数据文件包含 100 个 PDE，每个 PDE 提供 100 个样本，不同样本的初值 $g(x)$ 各不相同。
* 数据集路径：`custom/inverse`。
* 文件名格式：`custom_v4.23_inv_sinus0_[#bc]_f[#f]_c[#c]_k[#k]_seed[#r].hdf5`
    * 例如：`custom_v4.23_inv_sinus0_circ_fS_cU3_k1e-03_1_seed1.hdf5`
    * `[#f]`：PDE 形式中包含 $s(x),\kappa(x)$ 两项，该值指定这二者是否有空间依赖（即：是一个系数场）。例如，`S` 表示数据集里所有方程的 $s(x)$ 是系数场、$\kappa(x)=\kappa$ 为标量系数，`SK` 表示二者都是系数场，为空表示二者都是标量系数。
    * 其余可变项含义均与预训练数据集相同。
* 数据生成代码：[../data_generation/inverse_sinus.py](../data_generation/inverse_sinus.py)。

## 带三角函数项数据集

* 方程形式、边界条件：与预训练数据集类似，区别仅仅在于引入了三角函数项
    $f_i(u) = \sum_{k=1}^3c_{i0k}u^k + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$，$i=0,1$，
    其中 $h_{ij}\in\{\sin,\cos\}$ 等概率随机选取，$J_0+J_1=J$，其中 $J_0\in\{0,1,\dots,J\}$ 随机选取。
* 数据集路径：`custom/sinusJ`。
* 文件名格式：`custom_v4.23_sinus[#J]_[#bc]_c[#c]_k[#k]_seed[#r].hdf5`
    * 例如：`custom_v4.23_sinus4_circ_cU3_k1e-03_1_seed1.hdf5`
    * `[#J]`：PDE 中所含的三角函数项总数 $J$，取值为整数。
    * 其余可变项含义均与预训练数据集相同。
* 数据生成代码：[../data_generation/custom_sinus.py](../data_generation/custom_sinus.py)。

## 波方程数据集

* 方程形式：$u_{tt}+\mu u_t+Lu+bu_x+f(u)+s_T(t)s_X(x)=0$，$(t,x)\in[0,1]\times[-1,1]$，
    初值条件 $u(0,x)=g(x),u_t(0,x)=h(x)$。
    其中 $f(u) = c_{1}u+c_{2}u^2+c_{3}u^3$，
    $Lu$ 项从以下三种形式中等概率随机选取：$Lu=-c(x)^2u_{xx}$，$Lu=-c(x)(c(x)u_x)_x$，$Lu=-(c(x)^2u_x)_x$。
    源项 $s_T(t)s_X(x)$ 从零、标量系数、只有空间依赖、只有时间依赖、同时有时空依赖这 5 种情况中随机选取。
* 边界条件：周期+非周期。对于非周期边界情形，边界条件种类从 Dirichlet、Neumann、Robin 和 Mur（吸收边界条件）中随机选取，齐次与否也随机选取，且左右两端点的边界条件独立生成。
* 数据集路径：`custom/wave`。
* 文件名格式：`custom_v4.2_wave_[#bc]_c[#c]_k[#k]_seed[#r].hdf5`
    * 例如：`custom_v4.2_wave_interval_cU3_k1e-02_4_seed1.hdf5`
    * `[#bc]`：边界条件类型，`circ` 表示周期边界条件，`interval` 表示非周期边界条件。
    * `[#k]`：正系数 $\mu,c(x)^2$ 的范围。默认值 `1e-02_4`，表示其最小值为 $10^{-2}$、最大值为 4。
    * 其余可变项含义均与预训练数据集相同。
* 数据生成代码：[../data_generation/custom_wave.py](../data_generation/custom_wave.py)。
* 注意波方程数据集生成的过程中没有进行充分的筛选，因此有部分样本会包含由求解器产生的非物理模式（例如棋盘式的高频振荡）。
    我们用此验证 PDEformer 的训练过程能容忍一部分异常数据的存在，但如果用户希望将此数据用于其他目的，可能需要自行做一些额外的处理。

## 波方程反问题数据集

* 方程形式、边界条件与波方程数据集基本相同，主要区别在于要求 $s_T(t)s_X(x)$ 必须同时有时空依赖。
    对非周期边界情形，边界条件固定为左端点齐次 Mur 边界（波动传出），右端点齐次 Neumann 边界（无受力）。
    每个数据文件包含 100 个 PDE，每个 PDE 提供 100 个样本，不同样本的初值 $g(x),h(x)$ 和源项 $s_T(t)s_X(x)$ 各不相同。
* 数据集路径：`custom/inverse`。
* 文件名格式：`custom_v4.2_inv_wave_[#bc]_c[#c]_k[#k]_seed[#r].hdf5`
    * 例如：`custom_v4.2_inv_wave_interval_cU3_k1e-02_4_seed1.hdf5`
    * 各可变项含义均与波方程数据集相同。
* 数据生成代码：[../data_generation/inverse_wave.py](../data_generation/inverse_wave.py)。

## 多分量方程组数据集

* 方程形式：$\partial_tu_i + \sum_jc_{ij}u_j + s_i +\partial_x(\sum_ja_{ij}u_j + \sum_{j,k}b_{ijk}u_ju_k - \kappa_i\partial_xu_i) = 0$,
    其中 $0 \le i,j,k \le d-1$，$j \le k$，$(t,x)\in[0,1]\times[-1,1]$，$d$ 表示方程的变量（分量）个数。
    方程中出现的 $a_{ij},b_{ijk},c_{ij}$ 均为稀疏矩阵/稀疏张量，其中非零元素的个数从 $\{0,1,\dots,2d\}$ 中独立随机选取。
* 边界条件：为简便起见，只考虑周期边界条件。
* 数据集路径：`custom/mCompn`。
* 文件名格式：`custom_v4.2_compn[#d]_c[#c]_k[#k]_seed[#r].hdf5`
    * 例如：`custom_v4.2_compn2_cU3_k1e-03_1_seed1.hdf5`
    * `[#d]`：方程中分量个数 $d$，为正整数。
    * 其余可变项含义均与预训练数据集相同。
* 数据生成代码：[../data_generation/custom_multi_component.py](../data_generation/custom_multi_component.py)。
