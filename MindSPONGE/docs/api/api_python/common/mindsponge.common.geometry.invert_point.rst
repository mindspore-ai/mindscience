mindsponge.common.geometry.invert_point
=======================================

.. py:function:: mindsponge.common.geometry.invert_point(transformed_point, rotation, translation, extra_dims=0, stack=False, use_numpy=False)

    刚体群变换对点坐标的逆变换,即apply_to_point的逆变换。                                                     
    用旋转矩阵 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` 的转置矩阵和平移向量 :math:`(x, y, z)` translation 对坐标做旋转平移变化。                                                
    首先对初始坐标作平移变化,再将旋转矩阵 rotation 的转置矩阵与 rot_point 相乘得到最后坐标。

    .. math::
        rot_point = transformed_point - translation
        result = :math:`rotation^t` * rot_point

    其中向量的减法、转置与乘法具体过程可以参阅 vecs_sub、invert_rots、rots_mul_vecs等api。

    参数：
        - **transformed_point** (Tuple) - 输入的初始坐标，:math:`(x, y, z)`， 其中 x, y, z 均为 Tensor, 且 shape 相同。
        - **rotation** (Tuple) - 旋转矩阵，:math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`，且 xx、 xy等 均为 Tensor 且 shape 相同。
        - **translation** (Tuple) - 平移向量， :math:`(x, y, z)`, 其中 x, y, z 均为 Tensor, 且 shape 相同。
        - **extra_dims** (int) - 控制进行几层拓展。默认值：0。
        - **stack** (bool) - 控制是否进行入栈操作。默认值：False。
        - **use_numpy** (bool) - 控制是否使用 numpy。 默认值3：False。

    返回:
        - **invert_point** (Tuple) - 旋转平移后的坐标，长度为3。

