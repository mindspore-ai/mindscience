mindsponge.common.quaternion_from_tensor
========================================

.. py:function:: mindsponge.common.quaternion_from_tensor(tensor, normalize=False)

    利用输入的 `tensor` :math:`[(xx, xy, xz, yx, yy, yz, zz)]` ，进行仿射变换得到新的 `quaternion` ， `rotation`， `translation`。

    .. math::
        \begin{split}
        &tensor = [(xx, xy, xz, yx, yy, yz, zz)] \\
        &quaternion = (xx, xy, xz, yx) \\
        &translation = (yy, yz, zz) \\
        \end{split}


    再用生成的 `quaternion` 和 `translation` 进行仿射变换。仿射变换的过程参照 `quat_affine` API。

    参数：
        - **tensor** (Tensor) - 输入的初始Tensor :math:`[(xx, xy, xz, yx, yy, yz, zz)]` ，其中 :math:`[(xx, xy, xz, yx)]`
          与 `quaternion` 一致，:math:`(yy, yz, zz)` 与 `translation` 一致。
        - **normalize** (bool) - 控制是否要在仿射变换过程求范数。默认值： ``False``。

    返回：
        - **quaternion** (Tensor) - 四元数，shape为 :math:`(..., 4)` 的Tensor。
        - **rotation** (Tuple) - 旋转矩阵 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` ，且xx, xy等均为Tensor且shape相同。
        - **translation** (Tuple) - 平移向量 :math:`[(x, y, z)]` ，其中x, y, z均为Tensor，且shape相同。
