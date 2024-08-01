mindchemistry.e3.o3.matrix_to_angles
=========================================

.. py:function:: mindchemistry.e3.o3.matrix_to_angles(r_param)

    从矩阵到角度的转换。

    参数：
        - **R** (Tensor): 旋转矩阵。形状为 :math:`(..., 3, 3)` 的矩阵。

    返回：
        - **alpha** (Tensor) - Alpha 欧拉角。形状为 :math:`(...)` 的张量。
        - **beta** (Tensor) - Beta 欧拉角。形状为 :math:`(...)` 的张量。
        - **gamma** (Tensor) - Gamma 欧拉角。形状为 :math:`(...)` 的张量。

    异常：
        - **ValueError** - 如果 det(R) 不等于 1。


