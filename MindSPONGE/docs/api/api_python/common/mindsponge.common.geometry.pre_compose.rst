mindsponge.common.geometry.pre_compose
======================================

.. py:function:: mindsponge.common.geometry.pre_compose(quaternion, rotation, translation, update):

    利用 rotation 和 update 将输入的 quaternion 和 translation 进行更新，并用更新的 quaternion 和 translation
    进行新的仿射变化过程。

    得到更新的平移向量，旋转矩阵的过程如下：
    .. math::
        update = :math:`(xx, xy, xz, yx, yy, yz)`
        vector_quaternion_update = :math:`(xx, xy, xz)`
        x = :math:`(yx)`
        y = :math:`(yy)`
        z = :math:`(yz)`
        trans_update = :math:`[`(x, y, z)`]`
        new_quaternion = quaternion + vector_quaternion_update * quaternion
        rotated_trans_update = rotation * trans_update
        new_translation = translation + rotated_trans_update

    其中 vector_quaternion_update 与 quaternion 的相乘使用 quat_multiply_by_vec 函数相乘，
    rotation 与 trans_update的相乘用rots_mul_vecs函数，translation 与 rotated_trans_update 相加过程使用 vecs_add 函数。
    再用生成的 new_quaternion 和 new_translation 进行仿射变换。仿射变换的过程参照 quat_affine api。


    参数：
        - **quaternion** (Tensor) - 初始的待更新四元数， shape 为 :math:`[`(..., 4)`]` 的 Tensor。
        - **rotation** (Tuple) - 旋转矩阵，:math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`，且 xx、 xy等 均为 Tensor 且 shape 相同。
        - **translation** (Tuple) - 平移向量， :math:`(x, y, z)`, 其中 x, y, z 均为 Tensor, 且 shape 相同。
        - **update** (Tensor) - 用于辅助更新的矩阵， shape 为 :math:`[`(..., 6)`]` 的 Tensor，最后一维前三个元素为代表旋转矩阵的四元数三维向量表示（见quat_multiply_by_vec）


    返回:
        - **quaternion** (Tensor) - 更新后的四元数， shape 为 :math:`[`(..., 4)`]` 的 Tensor。
        - **rotation** (Tensor) - 更新后的旋转矩阵，:math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`，且 xx、 xy等 均为 Tensor 且 shape 相同。
        - **translation** (Tensor) - 更新后的平移向量， :math:`(x, y, z)`, 其中 x, y, z 均为 Tensor, 且 shape 相同。
