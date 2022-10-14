mindsponge.function.calc_angle_between_vectors
==============================================

.. py:class:: mindsponge.function.calc_angle_between_vectors(vector1, vector2)

    计算两个向量之间的角。

    参数：
        - **vector1** (Tensor) - 向量1，shape为(..., D)。
        - **vector2** (Tensor) - 向量2，shape为(..., D)。

    输出：
        Tensor。计算所得角。shape为(..., 1)。

    符号：
        - **D** - 模拟系统的维度。