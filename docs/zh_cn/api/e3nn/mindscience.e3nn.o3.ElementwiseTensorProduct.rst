mindscience.e3nn.o3.ElementwiseTensorProduct
======================================================

.. py:class:: mindscience.e3nn.o3.ElementwiseTensorProduct(irreps_in1, irreps_in2, filter_ir_out=None, ncon_dtype=mindspore.float32, **kwargs)

    元素级连接张量积。

    等效于 `TensorProduct` 的 `instructions='element'`。详细信息请参见 :class:`mindscience.e3nn.o3.TensorProduct`。

    参数：
        - **irreps_in1** (Union[str, Irrep, Irreps]) - 第一个输入的 Irreps。
        - **irreps_in2** (Union[str, Irrep, Irreps]) - 第二个输入的 Irreps。
        - **filter_ir_out** (Union[str, Irrep, Irreps, None], 可选) - 过滤器，仅选择特定的输出 `Irrep`。默认值：``None``。
        - **ncon_dtype** (mindspore.dtype, 可选) - ncon 计算模块输入张量的类型。默认值：``mindspore.float32``。
