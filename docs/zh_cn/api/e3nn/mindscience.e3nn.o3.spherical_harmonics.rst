mindscience.e3nn.o3.spherical_harmonics
=========================================

.. py:function:: mindscience.e3nn.o3.spherical_harmonics(l, x, normalize=True, normalization='integral')

    计算球面谐波。

    球面谐波是在三维空间上的多项式：

    .. math::
        Y^l: \mathbb{R}^3 \longrightarrow \mathbb{R}^{2l+1}

    通常限制在单位球面上（当 ``normalize=True``）：

    .. math::
        Y^l: S^2 \longrightarrow \mathbb{R}^{2l+1}

    满足以下性质：

    - 是笛卡尔坐标 ``x, y, z`` 的多项式；
    - 等变：:math:`Y^l(R x) = D^l(R) \, Y^l(x)`；
    - 正交：:math:`\int_{S^2} Y^l_m(x) \, Y^j_n(x) \, dx = \text{cste} \, \delta_{lj} \, \delta_{mn}`，常数取值依赖归一化选择。

    并满足如下递推性质：

    .. math::
        Y^{l+1}_i(x) = \text{cste}(l) \, C_{ijk} \, Y^l_j(x) \, x_k \\
        \partial_k Y^{l+1}_i(x) = \text{cste}(l) \, (l+1) \, C_{ijk} \, Y^l_j(x)

    其中 :math:`C` 为 `wigner_3j`。

    参数：
        - **l** (Union[int, list[int]]) - 球面谐波的阶数。
        - **x** (Tensor) - 构造球面谐波的张量，形状为 ``(..., 3)`` 的张量。
        - **normalize** (bool, 可选) - 是否在投影到球面谐波之前将 ``x`` 归一化为位于球面上的单位向量。默认值：``True``。
        - **normalization** (str, 可选) - {'integral', 'component', 'norm'}，输出张量的归一化方法。默认值：``'integral'``。

          - 'component'：:math:`\|Y^l(x)\|^2 = 2l+1, \; x \in S^2`
          - 'norm'：:math:`\|Y^l(x)\| = 1, \; x \in S^2`，即 ``component / sqrt(2l+1)``
          - 'integral'：:math:`\int_{S^2} Y^l_m(x)^2 \; dx = 1`，即 ``component / sqrt(4pi)``

    返回：
        Tensor，球面谐波 :math:`Y^l(x)`，张量形状为 ``(..., 2l+1)``。

    异常：
        - **ValueError** - 如果 `normalization` 不在 {'integral', 'component', 'norm'} 中。
        - **ValueError** - 如果 SphericalHarmonics 的 `irreps_in` 既不是向量 (`1x1o`) 也不是伪向量 (`1x1e`)。
        - **ValueError** - 如果球谐函数的 `irreps_out` 的 `l` 和 `p` 与 `irreps_in` 不一致，输出奇偶性应满足 `p = input_p**l`。
        - **ValueError** - 如果张量 `x` 的形状不是 ``(..., 3)``。
        - **NotImplementedError** - 如果 `l` 大于 11。