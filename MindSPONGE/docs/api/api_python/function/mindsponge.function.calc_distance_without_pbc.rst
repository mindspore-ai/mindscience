mindsponge.function.calc_distance_without_pbc
=============================================

.. py:class:: mindsponge.function.calc_distance_without_pbc(position_a, position_b, _pbc_box)

    在没有周期性边界条件的情况下计算位置A和B之间的距离。

    参数：
        - **position_a** (Tensor) - 位置A的坐标，shape为(..., D)。
        - **position_b** (Tensor) - 位置B的坐标，shape为(..., D)。
        - **_pbc_box** (None) - 虚假参数。

    输出：
        Tensor。计算所得距离。shape为(..., 1)。

    符号：
        - **D** - 模拟系统的维度。