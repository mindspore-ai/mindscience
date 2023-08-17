sponge.function.calc_distance_nopbc
=============================================

.. py:function:: sponge.function.calc_distance_nopbc(position_a, position_b, _pbc_box)

    在没有周期性边界条件的情况下计算位置A和B之间的距离，用绝对坐标计算。

    参数：
        - **position_a** (Tensor) - 位置A的坐标，shape为 :math:`(..., D)`。
        - **position_b** (Tensor) - 位置B的坐标，shape为 :math:`(..., D)`。
        - **_pbc_box** (None) - 默认值：None 不使用 pbc_box。

    输出：
        Tensor。A和B之间的距离。shape为 :math:`(..., 1)`。

    符号：
        - **D** - 模拟系统的空间维度, 一般为3。