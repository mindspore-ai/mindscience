mindchemistry.e3.o3.Norm
============================

.. py:class:: mindchemistry.e3.o3.Norm(irreps_in, squared=False, dtype=float32, ncon_dtype=float32)

    每一个Irrep在Irreps的直和中的范数。

    参数：
        - **irreps_in** (Union[str, Irrep, Irreps]) - 输入的Irreps。
        - **squared** (bool) - 是否返回平方范数。默认值: ``False``。
        - **dtype** (mindspore.dtype) - 输入张量的类型。默认值：``mindspore.float32`` 。
        - **ncon_dtype** (mindspore.dtype) - ncon 计算模块输入张量的类型。默认值：``mindspore.float32`` 。

    输入：
        - **v** (Tensor) - 形状为 ``(..., irreps_in.dim)`` 的张量。

    输出：
        - **output** (Tensor) - 形状为 ``(..., irreps_out.dim)`` 的张量。