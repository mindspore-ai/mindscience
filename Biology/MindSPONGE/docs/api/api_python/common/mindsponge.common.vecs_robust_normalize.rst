mindsponge.common.vecs_robust_normalize
========================================

.. py:function:: mindsponge.common.vecs_robust_normalize(v, epsilon=1e-8)

    向量l2范数归一化。

    .. math::
        \begin{split}
        &v=(x1,x2,x3) \\
        &l2\_norm=\sqrt{x1*x1+x2*x2+x3*x3+epsilon} \\
        &result=(x1/l2\_norm, x2/l2\_norm, x3/l2\_norm) \\
        \end{split}

    参数：
        - **v** (Tuple) - 输入向量，:math:`(x,y,z)` 其中 x,y,z 是标量或 Tensor，若为Tensor其shape相同。
        - **epsilon** (float) - 极小值，防止返回值为0，默认为 ``1e-8``。

    返回：
        Tuple， 返回二范数归一化后的向量，长度为3，其中每个元素shape与v中的元素相同。