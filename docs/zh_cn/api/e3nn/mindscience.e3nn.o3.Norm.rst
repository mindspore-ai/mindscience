mindscience.e3nn.o3.Norm
============================

.. py:class:: mindscience.e3nn.o3.Norm(irreps_in, squared=False, dtype=mindspore.float32, ncon_dtype=mindspore.float32)

    计算直和张量中每个不可约表示（irrep）的范数（长度）。
    给定一个在多个 irreps 的直和下变换的张量，本模块返回一个张量，其各个分量为每个 irrep 的范数。例如，若输入包含三个向量（`3x1o` irreps），则输出为一个 3 分量张量，其中每个分量为对应向量的欧氏范数。

    参数：
        - **irreps_in** (Union[str, Irrep, Irreps]) - 输入的 Irreps。
        - **squared** (bool, 可选) - 是否计算范数的平方。默认值：``False``。
        - **dtype** (mindspore.dtype, 可选) - 结果数据类型。默认值：``mindspore.float32``。
        - **ncon_dtype** (mindspore.dtype, 可选) - 用于ncon的数据类型。默认值：``mindspore.float32``。

    输入：
        - **v** (Tensor) - 形状为 :math:`(..., irreps\_in.dim)` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(..., irreps\_out.dim)` 的张量。
