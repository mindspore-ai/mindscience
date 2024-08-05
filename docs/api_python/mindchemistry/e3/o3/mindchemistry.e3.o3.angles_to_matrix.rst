mindchemistry.e3.o3.angles_to_matrix
=========================================

.. py:function:: mindchemistry.e3.o3.angles_to_matrix(alpha, beta, gamma)

    从角度到矩阵的转换。

    参数：
        - **alpha** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - alpha 欧拉角。形状为 :math:`(...)` 的张量。
        - **beta** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - beta 欧拉角。形状为 :math:`(...)` 的张量。
        - **gamma** (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]) - gamma 欧拉角。形状为 :math:`(...)` 的张量。


    返回：
        - **output** (Tensor) - 旋转矩阵。形状为 :math:`(..., 3, 3)` 的张量。
