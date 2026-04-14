sponge.function.gather_value
=================================

.. py:function:: sponge.function.gather_value(tensor: Tensor, index: Tensor)

    根据指标从张量的最后一根轴收集值。

    参数：
        - **tensor** (Tensor) - 输入张量，shape为 :math:`(B, X)` 。其中 :math:`B` 为batch size， :math:`X` 为输入张量第二维的任意大小。
        - **index** (Tensor) - 索引，shape为 :math:`(B, ...,)` 。

    返回：
        Tensor。取出的值，shape为 :math:`(B, ...,)` 。