sponge.function.squeeze_penulti
===================================

.. py:function:: sponge.function.squeeze_penulti(input_x: Tensor)

    返回从倒数第二个轴(axis=-2)删除大小为1的维度后的张量。

    参数：
        - **input_x** (Tensor) - 张量的shape为 :math:`(x_1, x_2, ..., x_{R-1}, x_R)`。

    返回：
        Tensor，张量的shape为 :math:`(x_1, x_2, ..., x_R)`。
