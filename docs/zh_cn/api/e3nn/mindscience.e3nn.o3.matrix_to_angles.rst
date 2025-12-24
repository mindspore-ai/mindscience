mindscience.e3nn.o3.matrix_to_angles
=========================================

.. py:function:: mindscience.e3nn.o3.matrix_to_angles(r_param)

    将 :math:`3 \times 3` 旋转矩阵转换为欧拉角 :math:`(\alpha, \beta, \gamma)`。

    参数：
        - **r_param** (Tensor) - 旋转矩阵。形状为 :math:`(..., 3, 3)` 的张量。

    返回：
        tuple[Tensor]，由 :math:`\alpha` 、:math:`\beta` 、:math:`\gamma` 组成的三元组。

    异常：
        - **ValueError** - 如果 det(R) 不等于 1。


