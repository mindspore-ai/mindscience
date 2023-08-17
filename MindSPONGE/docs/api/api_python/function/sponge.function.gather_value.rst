sponge.function.gather_value
=================================

.. py:function:: sponge.function.gather_value(tensor, index)

    根据指标从张量的最后一根轴收集值。

    参数：
        - **tensor** (Tensor) - 输入张量，shape为(B, X)。
        - **index** (Tensor) - 索引，shape为(B, ...,)。

    输出：
        Tensor。取出的值，shape为(B, ...,)。

    符号：
        - **B** - Batch size。
        - **X** - 输入张量第二维的任意大小。