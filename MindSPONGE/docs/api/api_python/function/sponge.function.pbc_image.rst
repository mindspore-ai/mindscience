sponge.function.pbc_image
==================================

.. py:function:: sponge.function.pbc_image(position, pbc_box, shift=0.0)

    计算PBC box的周期图。

    参数：
        - **position** (Tensor) - 位置坐标，数据类型为float，shape为(B, ..., D)。
        - **pbc_box** (Tensor) - PBC box，数据类型为float，shape为(B, D)。
        - **shift** (float) - PBC box的转换。默认值：0.0。

    输出：
        Tensor。周期图，shape为(B, ..., D)，数据类型为int32。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的维度，一般为3。