mindchemistry.e3.o3.xyz_to_angles
===================================

.. py:function:: mindchemistry.e3.o3.xyz_to_angles(xyz)

    将球体上的点 :math:`\vec r = (x, y, z)`转换为角度 :math:`(\alpha, \beta)`。

    .. math::
        \vec r = R(\alpha, \beta, 0) \vec e_z

    参数:
        - **xyz** (Tensor) - 点 :math:`(x, y, z)`。形状为 :math:`(..., 3)` 的张量。

    返回:
       - **alpha** (Tensor) - alpha 欧拉角。形状为:math:`(...)` 的张量。
       - **beta** (Tensor) - beta 欧拉角。形状为:math:`(...)` 的张量。
