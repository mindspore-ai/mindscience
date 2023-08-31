sponge.function.calc_torsion_by_vectors
============================================

.. py:function:: sponge.function.calc_torsion_by_vectors(vector1, vector2, axis_vector, keep_dims: bool = False)

    计算由三个向量形成的扭转角。

    参数：
        - **vector1** (Tensor) - 向量1，shape为 :math:`(..., D)` ，D是模拟系统的维度。
        - **vector2** (Tensor) - 向量2，shape为 :math:`(..., D)` 。
        - **axis_vector** (Tensor) - 轴向量，shape为 :math:`(..., D)` 。
        - **keepdims** (bool) - 设置为 ``True`` 的话，最后一个维度会保留，默认值 ``False`` 。

    输出：
        Tensor。计算所得扭转角。shape为 :math:`(...)` 或 :math:`(..., 1)` 。
