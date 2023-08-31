sponge.function.calc_distance_pbc
==========================================

.. py:function:: sponge.function.calc_distance_pbc(position_a, position_b, pbc_box, keepdims: bool = False)

    在有周期性边界条件的情况下计算位置A和B之间的距离，需要转化为同一个 pbc_box 内坐标计算 A 和B 的距离

    参数：
        - **position_a** (Tensor) - 位置A的坐标，shape为 :math:`(..., D)`，D是模拟系统的空间维度, 一般为3。
        - **position_b** (Tensor) - 位置B的坐标，shape为 :math:`(..., D)`。
        - **pbc_box** (Tensor) - 周期性盒子，shape为 :math:`(D)` 或 :math:`(B, D)`，B是Batch size。
        - **keepdims** (bool) - 设置为 ``True`` 的时候，最后一个维度会保留，默认值 ``False`` 。

    输出：
        Tensor。A和B之间的距离。shape为 :math:`(...)` 或 :math:`(..., 1)`。
