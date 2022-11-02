mindsponge.function.calc_distance_with_pbc
==========================================

.. py:function:: mindsponge.function.calc_distance_with_pbc(position_a, position_b, pbc_box)

    在有周期性边界条件的情况下计算位置A和B之间的距离。

    参数：
        - **position_a** (Tensor) - 位置A的坐标，shape为(B, ..., D)。
        - **position_b** (Tensor) - 位置B的坐标，shape为(B, ..., D)。
        - **pbc_box** (Tensor) - PBC box，shape为(B, D)。

    输出：
        Tensor。计算所得距离。shape为(B, ..., 1)。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的维度。