mindsponge.common.rots_expand_dims
==================================

.. py:function:: mindsponge.common.rots_expand_dims(rots, axis)

    对旋转矩阵 rots 的各个部分在指定的轴上添加额外维度。

    参数：
        - **rots** (Tuple) - 旋转矩阵 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` ，且xx, xy等均为Tensor且shape相同。
        - **axis** (Int) - 新插入的维度的位置，仅接受常量输入。

    返回：
        rots，Tuple。如果 axis 的值为0， xx 的 shape 为 :math:`(..., X_R)` ，其中 :math:`X_R` 为任意数，拓展后为 :math:`(1, ..., X_R)` ，若axis不为0则在对应轴拓展，返回拓展后的 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` 矩阵。