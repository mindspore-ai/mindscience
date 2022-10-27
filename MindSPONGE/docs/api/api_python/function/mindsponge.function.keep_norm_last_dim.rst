mindsponge.function.keep_norm_last_dim
======================================

.. py:class:: mindsponge.function.keep_norm_last_dim(vector)

    计算向量的归一化且保持最后输出的维度不变。

    参数：
        - **vector** (Tensor) - 输入的向量。

    输出：
        Tensor。归一化后的向量，shape(..., 1)。