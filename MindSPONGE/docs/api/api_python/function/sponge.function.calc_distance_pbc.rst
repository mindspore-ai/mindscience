sponge.function.calc_distance_pbc
==========================================

.. py:function:: sponge.function.calc_distance_pbc(position_a, position_b, pbc_box)

    在有周期性边界条件的情况下计算位置A和B之间的距离，需要转化为同一个 pbc_box 内坐标计算 A 和B 的距离

    参数：
        - **position_a** (Tensor) - 位置A的坐标，shape为 :math:`(B, ..., D)`。
        - **position_b** (Tensor) - 位置B的坐标，shape为 :math:`(B, ..., D)`。
        - **pbc_box** (Tensor) - 周期性盒子，shape为 :math:`(B, D)`。

    输出：
        Tensor。A和B之间的距离。shape为 :math:`(B, ..., 1)`。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的空间维度, 一般为3。