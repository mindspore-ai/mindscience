mindsponge.common.rots_mul_rots
===============================

.. py:function:: mindsponge.common.rots_mul_rots(x, y)

    获取两个旋转矩阵相乘结果，对目标进行两次旋转。

    .. math::
        \begin{split}
        &xx = xx1*xx2 + xy1*yx2 + xz1*zx2 \\
        &xy = xx1*xy2 + xy1*yy2 + xz1*zy2 \\
        &xz = xx1*xz2 + xy1*yz2 + xz1*zz2 \\
        &yx = yx1*xx2 + yy1*yx2 + yz1*zx2 \\
        &yy = yx1*xy2 + yy1*yy2 + yz1*zy2 \\
        &yz = yx1*xz2 + yy1*yz2 + yz1*zz2 \\
        &zx = zx1*xx2 + zy1*yx2 + zz1*zx2 \\
        &zy = zx1*xy2 + zy1*yy2 + zz1*zy2 \\
        &zz = zx1*xz2 + zy1*yz2 + zz1*zz2 \\
        \end{split}

    参数：
        - **x** (tuple) - 旋转矩阵x， :math:`(xx1, xy1, xz1, yx1, yy1, yz1, zx1, zy1, zz1)`。
        - **y** (tuple) - 旋转矩阵y， :math:`(xx2, xy2, xz2, yx2, yy2, yz2, zx2, zy2, zz2)`。

    返回：
        tuple，旋转矩阵x和旋转矩阵y的矩阵相乘结果, shape为 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`。
