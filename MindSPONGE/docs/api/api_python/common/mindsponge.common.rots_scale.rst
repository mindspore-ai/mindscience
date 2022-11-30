mindsponge.common.rots_scale
=============================

.. py:function:: mindsponge.common.rots_scale(rot, scale)

    对旋转矩阵进行缩放。

    .. math::
        \begin{split}
        &rot=(xx,xy,xz,yx,yy,yz,zx,zy,zz) \\
        &scaled\_{rots} = (scale*xx,scale*xy,scale*xz,scale*yx,scale*yy,scale*yz,scale*zx,scale*zy,scale*zz)
        \end{split}

    参数：
        - **rot** (Tuple) - 旋转矩阵，tuple长度为9，:math:`(xx,xy,xz,yx,yy,yz,zx,zy,zz)`，其中每个元素是标量或者Tensor。
          若为Tensor其shape相同。
        - **scale** (float) - 缩放值。

    返回：
        - **scaled_rots** (Tuple) -缩放后的旋转矩阵，tuple长度为9，shape与rot相同。