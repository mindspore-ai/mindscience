mindsponge.function.keep_norm_last_dim
======================================

.. py:class:: mindsponge.function.keep_norm_last_dim(vector)

    计算向量的归一化且保证最后的维度。

    参数：
        - **vector** (Tensor) - 输入的向量。

    输出：
        Tensor。向量，shape(..., 1)。