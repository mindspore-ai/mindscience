mindscience.e3nn.o3.FullyConnectedTensorProduct
======================================================

.. py:class:: mindscience.e3nn.o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out, ncon_dtype=mindspore.float32, **kwargs)

    全连接加权张量积。所有满足 :math:`|l_1 - l_2| \leq l_{out} \leq l_1 + l_2` 的路径都将被考虑。
    等效于 `TensorProduct` 的 `instructions='connect'`。详细信息请参见  :class:`mindscience.e3nn.o3.TensorProduct` 。

    参数：
        - **irreps_in1** (Union[str, Irrep, Irreps]) - 第一个输入的 Irreps。
        - **irreps_in2** (Union[str, Irrep, Irreps]) - 第二个输入的 Irreps。
        - **irreps_out** (Union[str, Irrep, Irreps]) - 输出的 Irreps。
        - **ncon_dtype** (mindspore.dtype, 可选) - ncon 计算模块输入张量的类型。默认值：``mindspore.float32``。
