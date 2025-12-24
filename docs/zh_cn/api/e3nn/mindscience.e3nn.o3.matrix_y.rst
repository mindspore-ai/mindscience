mindscience.e3nn.o3.matrix_y
============================

.. py:function:: mindscience.e3nn.o3.matrix_y(angle)

    返回绕 y 轴旋转给定角度的 :math:`3 \times 3` 旋转矩阵。

    参数：
        - **angle** (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]) - 围绕 y 轴的旋转角度。形状为 :math:`(...)`。

    返回：
        Tensor，绕 y 轴的旋转矩阵。输出形状为 :math:`(..., 3, 3)`。
