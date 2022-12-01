mindsponge.common.rigids_mul_rigids
===================================

.. py:function:: mindsponge.common.rigids_mul_rigids(a, b)

    把刚体变换 :math:`b` 从它所在局部坐标系中变换到刚体变换 :math:`a` 所在局部坐标系中。

    两个刚体变换的旋转矩阵相乘结果作为刚体变换 :math:`b` 新的旋转矩阵。

    用刚体变换 :math:`a` 的旋转矩阵与刚体变换 :math:`b` 的平移距离相乘，所得向量与刚体变换 :math:`a` 的平移距离相加，所得结果为刚体变换 :math:`b` 的新平移距离。

    .. math::
        \begin{split}
        &r = a_rb_r \\
        &t = a_rb_t +a_t \\
        \end{split}

    参数：
        - **a** (tuple) - 刚体变换 :math:`a` ，长度为2，包含旋转矩阵 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` 与平移距离 :math:`(x, y, z)` ，数据类型为标量或者shape相同的Tensor。
        - **b** (tuple) - 刚体变换 :math:`b` ，长度为2，包含旋转矩阵 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` 与平移距离 :math:`(x, y, z)` ，数据类型为标量或者shape相同的Tensor。

    返回：
        tuple(rots, trans)，变换后的刚体 :math:`b` ，长度为2，包含旋转矩阵 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` 与平移距离 :math:`(x, y, z)` ，数据类型为标量或者shape相同的Tensor。