mindsponge.function.displace_in_box
===================================

.. py:function:: mindsponge.function.displace_in_box(position, pbc_box, shift=0)

    在PBC box中展示系统的位置。

    参数：
        - **position** (Tensor) - 位置坐标，shape为(B, ..., D)。
        - **pbc_box** (Tensor) - PBC box，shape为(B, D)。
        - **shift** (float) - PBC box的转换。默认值：0。

    输出：
        Tensor。box中的位置。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的维度，shape为(B, ..., D)。