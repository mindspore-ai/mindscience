mindsponge.common.quat_affine
=============================

.. py:function:: mindsponge.common.quat_affine(quaternion, translation, rotation=None, normalize=True, unstack_inputs=False, use_numpy=False)

    基于旋转矩阵与平移向量生成仿射变换。

    参数：
        - **quaternion** (tensor) - shape为 :math:`(N_{res}, 4)` 。
        - **translation** (tensor) - shape为 :math:`(N_{res}, 3)` 。
        - **rotation** (tensor) - 旋转矩阵，shape为 :math:`(N_{res}, 9)` 。
        - **normalize** (bool) - 是否归一化，默认值： ``True``。
        - **unstack_inputs** (bool) - 输入为向量（``True``）还是张量（``False``），默认值： ``False``。
        - **use_numpy** (bool) - 是否使用numpy计算，默认值： ``False``。

    返回：
        返回仿射变换后结果。

        - 四元数 (tensor)，shape为 :math:`(N_{res}, 4)` 。
        - 旋转矩阵 (tuple) :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` ，每个元素shape为 :math:`(N_{res},)` 。
        - 平移向量 (tensor)，shape为 :math:`(N_{res}, 3)` 。