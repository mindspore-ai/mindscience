sponge.function.pbc_box_reshape
===================================

.. py:function:: sponge.function.pbc_box_reshape(pbc_box, ndim)

    把PBC box改变shape，使它的维度与ndim相同。

    参数：
        - **pbc_box** (Tensor) - 输入的PBC box。shape为(B, D)。
        - **ndim** (int) - PBC box的维度的数量。

    输出：
        Tensor。PBC box，shape为(B, 1, ..., 1, D)。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的维度。