sponge.function.calc_torsion_for_vectors
============================================

.. py:function:: sponge.function.calc_torsion_by_vectors(vector1, vector2, vector3)

    计算由三个向量形成的扭转角。

    参数：
        - **vector1** (Tensor) - 向量1，shape为(..., D)。
        - **vector2** (Tensor) - 向量2，shape为(..., D)。
        - **vector3** (Tensor) - 向量3，shape为(..., D)。

    输出：
        Tensor。计算所得扭转角。shape为(..., 1)。

    符号：
        - **D** - 模拟系统的维度。