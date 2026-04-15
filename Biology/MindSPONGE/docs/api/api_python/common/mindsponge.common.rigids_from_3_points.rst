mindsponge.common.rigids_from_3_points
======================================

.. py:function:: mindsponge.common.rigids_from_3_points(point_on_neg_x_axis, origin, point_on_xy_plane)

    Gram-Schmidt正交化过程。
    
    通过输入局部坐标系的原点O，在x轴负半轴上的点A以及x-y平面上一点P的坐标，计算该局部坐标系相对于全局坐标系的变换矩阵。

    首先根据三点坐标计算向量 :math:`\vec AO` 和 :math:`\vec OP` 的坐标，再根据 `rots_from_two_vecs` 得到的两个向量计算旋转矩阵。

    原点坐标到全局坐标系的原点坐标的距离为刚体的平移距离。

    最后返回旋转矩阵与平移距离。

    参考文献：
        `Jumper et al. (2021) Suppl. Alg. 21 'Gram-Schmidt process' <https://www.nature.com/articles/s41586-021-03819-2>`_。

    .. math::
        \begin{split}
        &\vec v_1 = \vec x_3 - \vec x_2 \\
        &\vec v_2 = \vec x_1 - \vec x_2 \\
        &\vec e_1 = \vec v_1 / ||\vec v_1|| \\
        &\vec u_2 = \vec v_2 - \vec  e_1(\vec e_1^T\vec v_2) \\
        &\vec e_2 = \vec u_2 / ||\vec u_2|| \\
        &\vec e_3 = \vec e_1 \times \vec e_2 \\
        &rotation = (\vec e_1, \vec e_2, \vec e_3) \\
        &translation = (\vec x_2) \\
        \end{split}

    参数：
        - **point_on_neg_x_axis** (tuple) - 在x轴负半轴上的点A的坐标，长度为3，数据类型为标量或者shape相同的Tensor。
        - **origin** (tuple) - 当前局部坐标系的原点O的坐标，长度为3，数据类型为标量或者shape相同的Tensor。
        - **point_on_xy_plane** (tuple) - x-y平面上一点P的坐标，长度为3，数据类型为标量或者shape相同的Tensor。

    返回：
        tuple(rots, trans)，长度为2，刚体的旋转矩阵 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` 与平移距离 :math:`(x, y, z)` ，数据类型为标量或者shape相同的Tensor。