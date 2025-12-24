mindscience.e3nn.o3.Linear
============================

.. py:class:: mindscience.e3nn.o3.Linear(irreps_in, irreps_out, ncon_dtype=mindspore.float32, **kwargs)

    线性等变操作。
    等效于 `TensorProduct` 的 `instructions='linear'`。详细信息请参见 :class:`mindscience.e3nn.o3.TensorProduct`。

    参数：
        - **irreps_in** (Union[str, Irrep, Irreps]) - 输入的 Irreps。
        - **irreps_out** (Union[str, Irrep, Irreps]) - 输出的 Irreps。
        - **ncon_dtype** (mindspore.dtype, 可选) - 用于ncon的数据类型。默认值：``mindspore.float32``。
