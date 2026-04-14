mindsponge.common.vecs_robust_norm
==================================

.. py:function:: mindsponge.common.vecs_robust_norm(v, epsilon=1e-8)

    求向量的l2范数。

    .. math::
        \begin{split}
        &v=(x1,x2,x3) \\
        &l2\_norm=\sqrt{x1*x1+x2*x2+x3*x3+epsilon} \\
        \end{split}

    参数：
        - **v** (Tuple) - 输入向量, :math:`(x,y,z)` 其中 x,y,z 是 标量或者Tensor，若为Tensor其shape相同。
        - **epsilon** (float) - 极小值，防止返回值为0，默认为 ``1e-8``。

    返回：
        Tensor，根据v计算得到的二范数, shape与v中的元素相同。