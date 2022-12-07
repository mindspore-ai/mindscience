mindsponge.common.initial_affine
================================

.. py:function:: mindsponge.common.initial_affine(num_residues, use_numpy=False)

    初始化仿射变换后的四元数，旋转矩阵，平移向量。

    参数：
        - **num_residues** (int) - 氨基酸残基数量。
        - **use_numpy** (bool) - 是否使用numpy计算，默认值：False。

    返回：
        返回初始化后仿射变换结果
        - 四元数 (tensor)，shape为 :math:`(N_{res}, 4)` 。
        - 旋转矩阵 (tuple) :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` ，每个元素shape为 :math:`(N_{res},)` 。
        - 平移向量 (tuple) :math:`(x, y, z)` ，每个元素shape为 :math:`(N_{res},)` 。