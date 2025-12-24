mindscience.e3nn.nn.OneHot
==========================

.. py:class:: mindscience.e3nn.nn.OneHot(num_types, dtype=mindspore.float32)

    支持不可约表示注解的 one-hot 嵌入。
    输出会使用 :class:`~.e3nn.o3.Irreps` 标注为标量（ :math:`l = 0` ）表示，可在期望 irreps 注解的 e3nn 网络中直接使用。

    参数：
        - **num_types** (int) - 不同原子类型的数量。
        - **dtype** (mindspore.dtype，可选) - 嵌入的数据类型。默认值：``mindspore.float32`` 。

    输入：
        - **atom_type** (Tensor) - 形状为 :math:`(...)` 的张量，包含整数的原子类型索引。

    输出：
        - **output** (Tensor) - 形状为 :math:`(..., \text{num\_types})` 的 one-hot 张量。