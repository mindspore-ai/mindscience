sponge.function.calc_vector_pbc
=======================================

.. py:function:: sponge.function.calc_vector_pbc(initial: Tensor, terminal: Tensor, pbc_box: Tensor)

    在有周期性边界条件的情况下，计算从起点到终点的向量。

    参数：
        - **initial** (Tensor) - 起点坐标，shape为 :math:`(..., D)` 。其中， :math:`D` 表示模拟系统的维度（通常为3）。
        - **terminal** (Tensor) - 终点坐标，shape为 :math:`(..., D)` 。
        - **pbc_box** (Tensor) - PBC box，shape为 :math:`(D)` 或 :math:`(B, D)` 。其中，:math:`B` 为batch size。

    返回：
        Tensor。计算所得向量。shape为 :math:`(..., D)`。