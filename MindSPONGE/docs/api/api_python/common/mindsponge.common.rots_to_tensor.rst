mindsponge.common.rots_to_tensor
================================

.. py:function:: mindsponge.common.rots_to_tensor(rots, use_numpy=False)

    将以向量表示的旋转矩阵转化为tensor， `rots_from_tensor` 的逆操作。

    参数：
        - **rots** (Tuple) - 使用向量表示的旋转矩阵， :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`。
        - **use_numpy** (bool) - 是否使用numpy计算，默认值：False。

    返回：
        tensor，最后一根轴合并后的旋转矩阵，shape为 :math:`(N_{res}, 3, 3)`。
