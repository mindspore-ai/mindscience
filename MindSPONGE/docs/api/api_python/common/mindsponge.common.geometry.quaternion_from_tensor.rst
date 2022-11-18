mindsponge.common.geometry.quaternion_from_tensor
=================================================

.. py:function:: mindsponge.common.geometry.quaternion_from_tensor(tensor, normalize=False):

    利用输入的 `tensor` ，:math:`[`(xx, xy, xz, yx, yy, yz, zz)`]` ,其中:math:`[`(xx, xy, xz, yx)`]`
    与 quaternion 一致，:math:`(yy, yz, zz)` 与 translation 一致，进行仿射变换得到新的 `quaternion`， `rotation`， `translation`。
    
    .. math::
        tensor = :math:`[`(xx, xy, xz, yx, yy, yz, zz)`]`
        quaternion = :math:`(xx, xy, xz, yx)`
        translation = :math:`(yy, yz, zz)`

    再用生成的 quaternion 和 translation 进行仿射变换。仿射变换的过程参照 quat_affine api。

    参数：
        - **tensor** (Tensor) - 输入的初始 Tensor ，:math:`[`(xx, xy, xz, yx, yy, yz, zz)`]` ,其中:math:`[`(xx, xy, xz, yx)`]`
        -                       与 quaternion 一致，:math:`(yy, yz, zz)` 与 translation 一致。
        - **normalize** (bool) - 控制是否要在仿射变换过程求范数。默认值： False


    返回:
        - **quaternion** (Tensor) - 四元数，shape 为 :math:`[`(..., 4)`]` 的 Tensor。
        - **rotation** (Tuple) - 旋转矩阵，:math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`，且 xx、 xy等均为 Tensor 且 shape 相同。
        - **translation** (Tuple) - 平移向量，:math:`[`(x, y, z)`]`, 其中 x, y, z 均为 Tensor, 且 shape 相同。
