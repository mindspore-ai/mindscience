sponge.function.pbc_box_reshape
===================================

.. py:function:: sponge.function.pbc_box_reshape(pbc_box: Tensor, ndim: int)

    把PBC box改变shape，使它的维度与ndim相同。

    参数：
        - **pbc_box** (Tensor) - 输入的PBC box。shape为 :math:`(B, D)` ，B是Batch size，D是模拟系统的维度。
        - **ndim** (int) - PBC box的维度的数量。

    返回：
        Tensor。PBC box，shape为 :math:`(B, 1, ..., 1, D)` 。
