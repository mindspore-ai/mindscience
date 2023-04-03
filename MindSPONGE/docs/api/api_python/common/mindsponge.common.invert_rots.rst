mindsponge.common.invert_rots
=============================

.. py:function:: mindsponge.common.invert_rots(m)

    输入一个旋转矩阵，输出旋转矩阵的转置矩阵。旋转矩阵 :math:`m = (xx, xy, xz, yx, yy, yz, zx, zy, zz)` ，则 :math:`m` 的转置为 :math:`m^{T} = (xx, yx, zx, xy, yy, zy, xz, yz, zz)`。

    参数：
        - **m** (tuple) - 旋转矩阵 :math:`m` ，长度为9，数据类型为标量或者shape相同的Tensor。

    返回：
        tuple， 旋转矩阵 :math:`m` 的逆，长度为9，数据类型为标量或者shape相同的Tensor。