mindscience.e3nn.o3.xyz_to_angles
===================================

.. py:function:: mindscience.e3nn.o3.xyz_to_angles(xyz)

    将单位球面上的点 :math:`\vec r = (x, y, z)` 转换为角度 :math:`(\alpha, \beta)`。

    .. math::
        \vec r = R(\alpha, \beta, 0) \vec e_z

    参数：
        - **xyz** (Tensor) - 点 :math:`(x, y, z)`。形状为 :math:`(..., 3)` 的张量。

    返回：
        tuple[Tensor]，由 :math:`\alpha`、:math:`\beta` 组成的二元组。
