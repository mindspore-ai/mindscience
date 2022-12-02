mindsponge.common.vecs_scale
=============================

.. py:function:: mindsponge.common.vecs_scale(v, scale)

    对向量的缩放。

    .. math::
        \begin{split}
        &v=(x1,x2,x3) \\
        &scaled\_{vecs} = (scale*x1,scale*x2,scale*x3) \\
        \end{split}

    参数：
        - **v** (Tuple) - 待缩放向量, :math:`(x,y,z)` 其中 x,y,z 是标量或者Tensor，若为Tensor其shape相同。
        - **scale** (float) - 缩放值。

    返回：
        Tuple，缩放后的向量，shape与v相同。