mindscience.e3nn.o3.FullTensorProduct
=========================================

.. py:class:: mindscience.e3nn.o3.FullTensorProduct(irreps_in1, irreps_in2, filter_ir_out=None, ncon_dtype=mindspore.float32, **kwargs)

    两个不可约表示之间的完全张量积。将生成所有可能的输出不可约表示，覆盖所有允许的组合 :math:`|l_1 - l_2| \leq l_{out} \leq l_1 + l_2`。
    等效于 `TensorProduct` 的 `instructions='full'`。详细信息请参见 :class:`mindscience.e3nn.o3.TensorProduct` 。

    参数：
        - **irreps_in1** (Union[str, Irrep, Irreps]) - 第一个输入的 Irreps。
        - **irreps_in2** (Union[str, Irrep, Irreps]) - 第二个输入的 Irreps。
        - **filter_ir_out** (Union[str, Irrep, Irreps, None], 可选) - 筛选输出中特定的 `Irrep`。默认值：``None``。
        - **ncon_dtype** (mindspore.dtype, 可选) - ncon 计算模块输入张量的类型。默认值：``mindspore.float32``。
