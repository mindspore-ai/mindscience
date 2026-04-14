mindsponge.common.quaternion_to_tensor
======================================

.. py:function:: mindsponge.common.quaternion_to_tensor(quaternion, translation)

    将输入的四元数变为Tensor。

    .. math::
        \begin{split}
        &quaternion = [(x_1, y_1, z_1, m_1)] \\
        &translation = [(x_2, y_2, z_2)] \\
        &result = [(x_1, y_1, z_1, m_1, x_2, y_2, z_2)] \\
        \end{split}
    
    参数：
        - **quaternion** (Tensor) - 输入的初始坐标，shape为 :math:`(..., 4)` 的Tensor。
        - **translation** (Tensor) - 坐标平移值，shape为 :math:`(..., 3)` 的Tensor。

    返回：
        Tensor，返回 `quaternion` 和 `translation` 的连接结果，shape为 :math:`(..., 7)` 。
