sponge.function.calc_vector_nopbc
==========================================

.. py:function:: sponge.function.calc_vector_nopbc(initial, terminal, _pbc_box)

    在没有周期性边界条件的情况下，计算从起点到终点的向量。

    参数：
        - **initial** (Tensor) - 起点坐标，shape为(B, ..., D)。
        - **terminal** (Tensor) - 终点坐标，shape为(B, ..., D)。
        - **_pbc_box** (None) - 虚假参数。

    输出：
        Tensor。计算所得向量。shape为(B, ..., D)。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的维度。