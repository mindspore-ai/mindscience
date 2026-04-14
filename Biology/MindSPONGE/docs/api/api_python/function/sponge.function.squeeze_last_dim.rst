sponge.function.squeeze_last_dim
====================================

.. py:function:: sponge.function.squeeze_last_dim(input_x: Tensor)

    返回从最后一个轴(axis=-1)删除大小为1的维度后的张量。

    参数：
        - **input_x** (Tensor) - 张量的shape为 :math:`(x_1, x_2, ..., x_R)`。

    返回：
        Tensor，张量的shape为 :math:`(x_1, x_2, ..., x_{R-1})`。
