mindsponge.common.invert_rigids
===============================

.. py:function:: mindsponge.common.invert_rigids(rigids)

    求解刚体变换的逆变换。
    
    通过 `invert_rots` 计算刚体变换的旋转矩阵的转置矩阵，再使用 `rots_mul_vecs` 让该旋转矩阵对平移距离进行旋转，所得平移距离的相反数即为刚体的逆平移距离。

    .. math::
        \begin{split}
        &inv\_rots = r_r^T = (r_0, r_3, r_6, r_1, r_4, r_7, r_2, r_5, r_8) \\
        &inv\_trans = -r_r^T \cdot r_t^T = (- (r_0 \times t_0 + r_3 \times t_0 + r_6 \times t_0), -(r_1 \times t_1 + r_4 \times t_1 + r_7 \times t_1), -(r_2 \times t_2 + r_5 \times t_2 + r_8 \times t_2)) \\
        \end{split}

    参数：
        - **rigids** (tuple) - 把刚体从当前坐标系仿射变换到另一个坐标系的旋转矩阵与平移矩阵。

    返回：
        tuple(rots, trans)。把刚体坐标从当前坐标系变换到另一个坐标系的旋转矩阵与平移矩阵，长度为2，包含旋转矩阵 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` 与平移距离 :math:`(x, y, z)` ，数据类型为标量或者shape相同的Tensor。