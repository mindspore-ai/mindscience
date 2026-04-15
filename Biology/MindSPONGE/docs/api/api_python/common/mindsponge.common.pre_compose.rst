mindsponge.common.pre_compose
=============================

.. py:function:: mindsponge.common.pre_compose(quaternion, rotation, translation, update)

    利用旋转矩阵rotation和辅助矩阵update更新输入的四元数quaternion和平移向量translation，并进行新的仿射变化过程，得到更新的平移向量。
    
    旋转矩阵的过程如下：

    .. math::
        \begin{split}
        &update = (xx, xy, xz, yx, yy, yz) \\
        &vector\_quaternion\_update = (xx, xy, xz) \\
        &x = (yx) \\
        &y = (yy) \\
        &z = (yz) \\
        &trans\_update = [(x, y, z)] \\
        &new\_quaternion = quaternion + vector\_quaternion\_update * quaternion \\
        &rotated\_trans\_update = rotation * trans\_update \\
        &new\_translation = translation + rotated\_trans\_update \\
        \end{split}

    其中 `vector_quaternion_update` 与 `quaternion` 的相乘使用 `quat_multiply_by_vec` 函数相乘，
    `rotation` 与 `trans_update` 的相乘用 `rots_mul_vecs` 函数， `translation` 与 `rotated_trans_update` 相加过程使用 `vecs_add` 函数。
    再用生成的 `new_quaternion` 和 `new_translation` 进行仿射变换。仿射变换的过程参照 `quat_affine` API。

    参数：
        - **quaternion** (Tensor) - 初始的待更新四元数，shape为 :math:`[(..., 4)]` 的Tensor。
        - **rotation** (Tuple) - 旋转矩阵 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` ，且xx, xy等均为Tensor且shape相同。
        - **translation** (Tuple) - 平移向量 :math:`(x, y, z)` ，其中x, y, z均为Tensor，且shape相同。
        - **update** (Tensor) - 用于辅助更新的矩阵，shape为 :math:`[(..., 6)]` 的Tensor，最后一维前三个元素为代表旋转矩阵的四元数三维向量表示，参考 `quat_multiply_by_vec` 。

    返回：
        - **quaternion** (Tensor) - 更新后的四元数，shape为 :math:`[(..., 4)]` 的Tensor。
        - **rotation** (Tensor) - 更新后的旋转矩阵 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` ，且xx, xy等均为Tensor且shape相同。
        - **translation** (Tensor) - 更新后的平移向量 :math:`(x, y, z)` ，其中x, y, z均为Tensor，且shape相同。
