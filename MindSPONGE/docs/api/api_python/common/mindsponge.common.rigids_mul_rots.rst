mindsponge.common.rigids_mul_rots
=================================

.. py:function:: mindsponge.common.rigids_mul_rots(x, y)

    使用旋转矩阵 :math:`\vec y` 对刚体变换 :math:`x` 进行旋转。
    
    即使用 `rots_mul_rots` 让旋转矩阵 :math:`\vec y` 与刚体的旋转矩阵 :math:`x[0]` 相乘，平移距离不发生变化。

    .. math::
        (r, t) = (x_ry, x_t)

    参数：
        - **x** (tuple) - 刚体变换 :math:`x` ，长度为2，包含旋转矩阵 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` 与平移距离 :math:`(x, y, z)` ，数据类型为标量或者shape相同的Tensor。
        - **y** (tuple) - 旋转矩阵 :math:`\vec y` ，长度为9，数据类型为标量或者shape相同的Tensor。

    返回：
        tuple(rots, trans)，长度为2，包含刚体进一步旋转后的旋转矩阵与未发生变化的平移距离。