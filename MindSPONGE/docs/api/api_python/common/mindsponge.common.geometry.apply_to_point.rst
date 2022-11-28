mindsponge.common.geometry.apply_to_point
=========================================

.. py:function:: mindsponge.common.geometry.apply_to_point(rotation, translation, point, extra_dims=0)

    对输入坐标进行旋转平移变换。

    .. math::
        rot_point = rotation * point
        result = rot_point + translation

    具体的乘法过程与加法过程可以参考 rots_mul_vecs 和 vecs_add api。

    参数：
        - **rotation** (Tuple) - 旋转矩阵，:math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`，且 xx、 xy等 均为 Tensor 且 shape 相同。
        - **translation** (Tuple) - 平移向量， :math:`[(x, y, z)]`，其中 x, y, z 均为 Tensor, 且 shape 相同。
        - **point** (Tensor) - 初始坐标值， :math:`[(x, y, z)]`，其中 x, y, z 均为 Tensor, 且 shape 相同。
        - **extra_dims** (Int) - 控制进行几层拓展。默认值： 0

    返回:
        - **result** (Tuple) - 转化后的坐标，长度为3。
