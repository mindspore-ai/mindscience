mindscience.e3nn.o3.TensorSquare
=================================

.. py:class:: mindscience.e3nn.o3.TensorSquare(irreps_in, irreps_out=None, filter_ir_out=None, ncon_dtype=mindspore.float32, **kwargs)

    计算张量的平方张量积。
    等价于 `TensorProduct` 使用 `irreps_in2=None` 且 `instructions='full'` 或 `'connect'`。详细信息见 :class:`mindscience.e3nn.o3.TensorProduct`。
    如果提供了 `irreps_out`，此操作将是完全连接的。如果未提供 `irreps_out`，则此操作没有参数，类似于完全张量积。

    参数：
        - **irreps_in** (Union[str, Irrep, Irreps]) - 输入的 Irreps。
        - **irreps_out** (Union[str, Irrep, Irreps, None], 可选) - 输出的 Irreps。默认值: ``None``。
        - **filter_ir_out** (Union[str, Irrep, Irreps, None], 可选) - 过滤器，用于仅选择特定的 `Irrep` 输出。默认值: ``None``。
        - **ncon_dtype** (mindspore.dtype, 可选) - `ncon` 计算模块的输入张量的类型。默认值: ``mindspore.float32``。

    异常：
        - **ValueError** - 如果 `irreps_out` 和 `filter_ir_out` 都不为 None。
