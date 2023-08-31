sponge.function.pbc_image
==================================

.. py:function:: sponge.function.pbc_image(position, pbc_box, offset=0.0)

    计算PBC box的周期图。

    参数：
        - **position** (Tensor) - 位置坐标，数据类型为float，shape为 :math:`(B, ..., D)` ，B是Batch size，D是模拟系统的维度，一般为3。
        - **pbc_box** (Tensor) - PBC box，数据类型为float，shape为 :math:`(B, D)` 。
        - **offset** (float) - 偏移率。默认值： ``0.0`` 。

    输出：
        Tensor。周期图，shape为 :math:`(B, ..., D)` ，数据类型为int32。
