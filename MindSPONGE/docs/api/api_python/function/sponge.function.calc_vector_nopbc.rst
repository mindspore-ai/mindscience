sponge.function.calc_vector_nopbc
==========================================

.. py:function:: sponge.function.calc_vector_nopbc(initial, terminal)

    在没有周期性边界条件的情况下，计算从起点到终点的向量。

    参数：
        - **initial** (Tensor) - 起点坐标，shape为 :math:`(..., D)` 。其中， :math:`D` 表示模拟系统的维度（通常为3）。
        - **terminal** (Tensor) - 终点坐标，shape为 :math:`(..., D)` 。

    返回：
        Tensor。计算所得向量。shape为 :math:`(..., D)`。