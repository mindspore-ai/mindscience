mindsponge.common.vecs_sub
===========================

.. py:function:: mindsponge.common.vecs_sub(v1, v2)

    对两个向量执行减法操作。

    .. math::
        \begin{split}
        &v1=(x1,x2,x3) \\
        &v2=(x1',x2',x3') \\
        &result=(x1-x1',x2-x2',x3-x3') \\
        \end{split}

    参数：
        - **v1** (Tuple) - 输入向量1 :math:`(x, y, z)` 其中x, y, z为标量或Tensor，若为Tensor其shape相同。
        - **v2** (Tuple) - 输入向量2 :math:`(x, y, z)` 其中x, y, z为标量或Tensor，若为Tensor其shape相同。

    返回：
        Tuple，长度为3， :math:`(x', y', z')` ，其中x', y', z'为Tensor，且shape与v1相同。