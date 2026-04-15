mindsponge.common.vecs_expand_dims
==================================

.. py:function:: mindsponge.common.vecs_expand_dims(v, axis)

    将输入的v在指定的轴添加额外维度。

    参数：
        - **v** (Tuple) - 输入的初始向量，长度为3，:math:`(xx, xy, xz)`。
        - **axis** (int) - 新插入的维度的位置,仅接受常量输入。

    返回：
        tuple，如果 axis 的值为0，且 :math:`xx` 的shape为 :math:`(..., X_R)` ，拓展后的shape为 :math:`(1, ..., X_R)` 。若 axis为其它值，则在其它方向拓展，返回拓展后的 :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` 。