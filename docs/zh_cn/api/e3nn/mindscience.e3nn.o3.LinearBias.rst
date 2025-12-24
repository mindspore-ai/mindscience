mindscience.e3nn.o3.LinearBias
================================

.. py:class:: mindscience.e3nn.o3.LinearBias(irreps_in, irreps_out, has_bias, ncon_dtype=mindspore.float32, **kwargs)

    具有添加偏置选项的线性等变操作。
    等效于 `TensorProduct` 的 `instructions='linear'`，并可选择添加偏置。详细信息请参见 :class:`mindscience.e3nn.o3.TensorProduct`。

    参数：
        - **irreps_in** (Union[str, Irrep, Irreps]) - 输入的Irreps。
        - **irreps_out** (Union[str, Irrep, Irreps]) - 输出的Irreps。
        - **has_bias** (bool) - 是否将偏差添加到计算中。
        - **ncon_dtype** (mindspore.dtype, 可选) - ncon 计算模块输入张量的类型。默认值：``mindspore.float32`` 。

    输入：
        - **v1** (Tensor) - 输入张量，形状为 :math:`(..., 2l+1)`。
        - **v2** (Tensor, 可选) - 输入张量，形状为 :math:`(..., 2l+1)`。默认值：``None``。
        - **weight** (Tensor, 可选) - 权重张量，形状为 :math:`(..., 2l+1)`。默认值：``None``。

    输出：
        - **out** (Tensor) - 输出张量，形状为 :math:`(..., 2l+1)`。
