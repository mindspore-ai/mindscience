mindsponge.function.norm_last_dim
=================================

.. py:function:: mindsponge.function.norm_last_dim(vector)

    计算向量的归一化且删除最后的维度。

    参数：
        - **vector** (Tensor) - 输入的向量，shape为(..., D)。

    输出：
        Tensor。向量，shape(...,)。

    符号：
        - **D** - 模拟系统的维度。