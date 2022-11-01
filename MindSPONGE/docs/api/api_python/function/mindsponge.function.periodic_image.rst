mindsponge.function.periodic_image
==================================

.. py:function:: mindsponge.function.periodic_image(position, pbc_box, shift=0)

    计算PBC box的周期图。

    参数：
        - **position** (Tensor) - 位置坐标，shape为(B, ..., D)。
        - **pbc_box** (Tensor) - PBC box，shape为(B, D)。
        - **shift** (float) - PBC box的转换。默认值：0。

    输出：
        Tensor。周期图，shape为(B, ..., D)。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的维度。