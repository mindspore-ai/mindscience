sponge.function.coordinate_in_pbc
=====================================

.. py:function:: sponge.function.coordinate_in_pbc(position: Tensor, pbc_box: Tensor, offset: float = 0)

    在主PBC box中获取坐标

    参数：
        - **position** (Tensor) - 张量的shape为 :math:`(B, ..., D)`。数据类型是float。位置坐标 :math:`R`
          其中，B是Batchsize。D是模拟系统的空间维度，通常为3。
        - **pbc_box** (Tensor) - 张量的shape为 :math:`(B, D)`。数据类型是float。PBC box的大小 :math:`\vec{L}`
        - **offset** (float) - 偏移比 :math:`c` 与box的大小 :math:`\vec{L}` 的相对距离。默认值：0.0

    返回：
        Tensor。坐标。张量的shape为 :math:`(B, ..., D)`。数据类型是float。

