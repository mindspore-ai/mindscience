mindscience.e3nn.o3.SphericalHarmonics
=========================================

.. py:class:: mindscience.e3nn.o3.SphericalHarmonics(irreps_out, normalize, normalization='integral', irreps_in=None, dtype=mindspore.float32)

    将三维笛卡尔向量 `(x, y, z)` 映射到复值球面谐波基函数 :math:`Y_l^m(\hat{x})` 的层。可返回任意指定阶数 `l`，并自动处理奇偶性（偶/奇）选择规则。

    参数：
        - **irreps_out** (Union[str, `Irrep`, `Irreps`]) - 球面谐波基函数输出的不可约表示。
        - **normalize** (bool) - 如果为 ``True``，则范数使得 :math:`\int |Y|^2 = 1`。注意：现在强制要求 ``normalize=True``。
        - **normalization** (str, 可选) - {'integral', 'component', 'norm'}，归一化方式。默认值：``'integral'``。
        - **irreps_in** (Union[str, `Irrep`, `Irreps`], 可选) - 输入的 Irreps。默认值：``None``。
        - **dtype** (mindspore.dtype, 可选) - 结果数据类型。默认值：``mindspore.float32``。

    输入：
        - **x** (Tensor) - 构造球面谐波的张量。形状为 :math:`(..., 3)` 的张量。

    输出：
        - **output** (Tensor) - 球面谐波 :math:`Y^l(x)`，张量形状为 :math:`(..., 2l+1)`。

    异常：
        - **ValueError** - 如果 `normalization` 不在 {'integral', 'component', 'norm'} 中。
        - **ValueError** - 如果球谐函数的 `irreps_in` 既不是向量 (`1x1o`) 也不是伪向量 (`1x1e`)。
        - **ValueError** - 如果球谐函数的 `irreps_out` 的 `l` 和 `p` 与 `irreps_in` 不一致。输出的奇偶性应为 p = {input_p**l}。
        - **NotImplementedError** - 如果 `l` 大于 11。