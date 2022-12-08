mindsponge.common.make_transform_from_reference
===============================================

.. py:function:: mindsponge.common.make_transform_from_reference(point_a, point_b, point_c)

    使用施密特正交化方法构造骨架的旋转矩阵和平移向量。

    计算旋转矩阵和平移满足

    a）'N'原子是原始点

    b）'CA'原子位于x轴上
    
    c）平面CA-N-C在x-y平面上。

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
        - **point_a** (float, tensor) -> (tensor) - 'N'原子空间位置信息，shape为: :math:`[..., N_{res}, 3]` 。
        - **point_b** (float, tensor) -> (tensor) - 'CA'原子空间位置信息，shape为: :math:`[..., N_{res}, 3]` 。
        - **point_c** (float, tensor) -> (tensor) - 'C'原子空间位置信息，shape为: :math:`[..., N_{res}, 3]` 。

    返回：
        旋转矩阵(tuple) :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` ，每个元素shape为 :math:`(..., N_{res})` 。
        平移向量(tuple) :math:`(x, y, z)` 每个元素shape为 :math:`(..., N_{res})` 。