sponge.function.gather_vector
==================================

.. py:function:: sponge.function.gather_vector(tensor, index)

    根据指标从张量的倒数第二轴收集向量。

    参数：
        - **tensor** (Tensor) - 输入张量，shape为 :math:`(B, X, D)` 。其中 :math:`B` 为batch size， :math:`X` 为任意大小， :math:`D` 为模拟系统的维度，通常为3。
        - **index** (Tensor) - 索引，shape为 :math:`(B, ...,)`。

    返回：
        Tensor。取出的向量。shape为 :math:`(B, ..., D)`。