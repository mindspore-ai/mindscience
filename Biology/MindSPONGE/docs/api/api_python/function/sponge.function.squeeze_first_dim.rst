sponge.function.squeeze_first_dim
=====================================

.. py:function:: sponge.function.squeeze_first_dim(input_x: Tensor)

    返回从第一个轴(axis=0)删除大小为1的维度后的张量。

    参数：
        - **input_x** (Tensor) - 张量的shape为 :math:`(x_1, x_2, ..., x_R)`。

    返回：
        Tensor，张量的shape为 :math:`(x_2, ..., x_R)`。
