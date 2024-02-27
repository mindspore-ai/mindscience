mindsponge.common.apply_to_point
=========================================

.. py:function:: mindsponge.common.apply_to_point(rotation, translation, point, extra_dims=0)

    对输入坐标进行旋转平移变换。

    .. math::
        \begin{split}
        &rot\_point = rotation \cdot point \\
        &result = rot\_point + translation \\
        \end{split}

    具体的乘法过程与加法过程可以参考 `mindsponge.common.rots_mul_vecs` 和 `mindsponge.common.vecs_add` API。

    参数：
        - **rotation** (Tuple) - 旋转矩阵，:math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` ，且 :math:`xx, xy` 等均为Tensor且shape相同。
        - **translation** (Tuple) - 平移向量， :math:`[(x, y, z)]` ，其中 :math:`x, y, z` 均为Tensor，且shape相同。
        - **point** (Tensor) - 初始坐标值， :math:`[(x, y, z)]` ，其中 :math:`x, y, z` 均为Tensor，且shape相同。
        - **extra_dims** (int) - 控制进行几次维度的拓展。默认值： ``0``。

    返回：
        Tuple，转化后的坐标，长度为3。