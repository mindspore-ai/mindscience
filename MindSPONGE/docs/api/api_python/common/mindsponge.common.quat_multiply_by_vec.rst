mindsponge.common.quat_multiply_by_vec
======================================

.. py:function:: mindsponge.common.quat_multiply_by_vec(quat, vec)

    计算四元数与纯向量四元数的乘积。

    .. math::
        \begin{split}
        &temp =  QUAT\_MULTIPLY\_BY\_VEC * quat[..., :, None, None] * vec[..., None, :, None] \\
        &result = sum(temp,axis=(-3, -2)) \\
        \end{split}

    参数：
        - **quat** (Tensor) - 输入的四元数，shape为 :math:`(..., 4)` 的 Tensor。
        - **vec** (Tensor) - 纯向量四元数， :math:`(b, c, d)` 的未归一化四元数,其中归一化四元数能被表示成 :math:`(1, b, c, d)`。

    返回：
        Tensor，计算后的结果，shape 为 :math:`(..., 4)` 。