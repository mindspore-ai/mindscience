mindsponge.function.gather_values
=================================

.. py:class:: mindsponge.function.gather_values(tensor, index)

    根据指标从张量的最后一根轴收集值。

    参数：
        - **tensor** (Tensor) - 输入张量，shape为(B, X)。
        - **index** (Tensor) - 索引，shape为(B, ...,)。

    输出：
        Tensor。取出的值。