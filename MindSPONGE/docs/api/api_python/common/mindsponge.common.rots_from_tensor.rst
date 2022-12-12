mindsponge.common.rots_from_tensor
==================================

.. py:function:: mindsponge.common.rots_from_tensor(rots, use_numpy=False)

    输入tensor，将最后两根轴对应的3*3的旋转矩阵摊平拆分，得到旋转矩阵的每个分量，rots_to_tensor的逆操作。

    参数：
        - **rots** (Tensor) - 代表旋转矩阵，shape为 :math:`(..., 3, 3)` 。
        - **use_numpy** (bool) - 是否使用numpy计算，默认值：False。

    返回：
        tuple，使用向量表示的旋转矩阵，shape为 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` 。
