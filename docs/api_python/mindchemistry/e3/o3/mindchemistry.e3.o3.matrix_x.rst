mindchemistry.e3.o3.matrix_x
============================

.. py:function:: mindchemistry.e3.o3.matrix_x(angle)

    给出给定角度下绕x轴的旋转矩阵。

    参数:
        - **angle** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - 围绕x轴的旋转角度。形状为:math:`(...)`。

    返回:
        - **output** (Tensor) - 围绕x轴的旋转矩阵。输出的形状为 :math:`(..., 3, 3)`。
