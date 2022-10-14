mindsponge.function.gather_vectors
==================================

.. py:class:: mindsponge.function.gather_vectors(tensor, index)

    根据指标从张量的倒数第二轴收集向量。

    参数：
        - **tensor** (Tensor) - 输入张量，shape为(B, A, D)。
        - **index** (Tensor) - 索引，shape为(B, ...,)。

    输出：
        Tensor。取出的向量。