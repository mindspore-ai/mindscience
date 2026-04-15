mindsponge.common.quat_to_rot
=============================

.. py:function:: mindsponge.common.quat_to_rot(normalized_quat, use_numpy=False)

    将四元数转化为旋转矩阵。

    .. math::
        \begin{split}
        &xx = 1 - 2 * y * y - 2 * z * z \\
        &xy = 2 * x * y + 2 * w * z \\
        &xz = 2 * x * z - 2 * w * y \\
        &yx = 2 * x * y - 2 * w * z \\
        &yy = 1 - 2 * x * x - 2 * z * z \\
        &yz = 2 * z * y + 2 * w * x \\
        &zx = 2 * x * z + 2 * w * y \\
        &zy = 2 * y * z - 2 * w * x \\
        &zz = 1 - 2 * x * x - 2 * y * y \\
        \end{split}

    参数：
        - **normalized_quat** (tensor) - 归一化的四元数，shape为 :math:`(N_{res}, 4)` 。
        - **use_numpy** (bool) - 是否使用numpy计算，默认值： ``False``。

    返回：
        旋转矩阵(tuple)， :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` , 每个元素shape :math:`(N_{res}, )` 。
