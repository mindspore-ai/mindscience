sponge.function.calc_vector
==============================

.. py:function:: sponge.function.calc_vector(initial, terminal, pbc_box)

    计算从起点到终点的向量。

    参数：
        - **initial** (Tensor) - 起点坐标，shape为(B, ..., D)。
        - **terminal** (Tensor) - 终点坐标，shape为(B, ..., D)。
        - **pbc_box** (Tensor) - PBC box，shape为(B, D)。

    输出：
        Tensor。计算所得向量。shape为(B, ..., D)。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的维度。